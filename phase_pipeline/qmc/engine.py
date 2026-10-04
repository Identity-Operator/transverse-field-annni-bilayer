"""PIMC engine: discrete imaginary-time worldline Metropolis.

A Numba implementation is used for production; all replicas are independent
and are parallelized across CPU threads.
"""
from __future__ import annotations

import json
import math

import numpy as np

try:
    from numba import njit, prange, set_num_threads
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco
    prange = range
    def set_num_threads(n):
        return None

from ..io_utils import autocorr_time


@njit(cache=True)
def _xorshift64star(state):
    x = state
    x ^= (x >> np.uint64(12))
    x ^= (x << np.uint64(25))
    x ^= (x >> np.uint64(27))
    state = x
    rnd = (x * np.uint64(2685821657736338717))
    # 53 random bits -> [0,1)
    u = float((rnd >> np.uint64(11)) & np.uint64((1 << 53) - 1)) / float(1 << 53)
    return state, u


@njit(cache=True)
def _initial_pattern(S, q, L, Mtau, init_code, rng_state):
    # init_code: 0 random, 1 FM, 2 A, 3 MOD(++--), 4 C(+-- along y)
    for l in range(2):
        for x in range(L):
            for y in range(L):
                if init_code == 1:
                    base = 1
                elif init_code == 2:
                    base = 1 if l == 0 else -1
                elif init_code == 3:
                    base = 1 if (y % 4) < 2 else -1
                    if l == 1:
                        base = base
                elif init_code == 4:
                    base = 1 if (y % 2) == 0 else -1
                else:
                    rng_state, u = _xorshift64star(rng_state)
                    base = 1 if u < 0.5 else -1
                for t in range(Mtau):
                    S[q, l, x, y, t] = base
    return rng_state


@njit(cache=True)
def _measure_replica(s, L, Mtau, beta, Gamma):
    # Returns local correlations, template moments, q-spectrum and energy pieces.
    nsite = 2 * L * L
    cx = cy = c2 = cp = cdiag = cx2 = 0.0
    mfm_acc = ma_acc = mc_acc = mg_acc = 0.0
    mfm2 = ma2 = mc2 = mg2 = 0.0
    mfm4 = ma4 = mc4 = mg4 = 0.0
    abs_mfm = abs_ma = abs_mc = abs_mg = 0.0
    temporal = 0.0
    # q spectrum accumulated as |m_l(q)|^2 / N, equal-time averaged over tau.
    Sq = np.zeros(L, dtype=np.float64)
    # We also accumulate complex q amplitudes moments for binder-like diagnostics.
    q2 = np.zeros(L, dtype=np.float64)
    q4 = np.zeros(L, dtype=np.float64)

    for t in range(Mtau):
        sfm = sa = sc = sg = 0.0
        for l in range(2):
            for x in range(L):
                for y in range(L):
                    v = float(s[l, x, y, t])
                    sfm += v
                    sa += v * (1.0 if l == 0 else -1.0)
                    sc += v * (1.0 if (y % 2) == 0 else -1.0)
                    sg += v * (1.0 if ((y + l) % 2) == 0 else -1.0)
                    cx += v * s[l, (x + 1) % L, y, t]
                    cy += v * s[l, x, (y + 1) % L, t]
                    c2 += v * s[l, x, (y + 2) % L, t]
                    cdiag += v * s[l, (x + 1) % L, (y + 1) % L, t]
                    cx2 += v * s[l, (x + 2) % L, y, t]
                    temporal += v * s[l, x, y, (t + 1) % Mtau]
        for x in range(L):
            for y in range(L):
                cp += s[0, x, y, t] * s[1, x, y, t]
        mf = sfm / nsite
        maa = sa / nsite
        mcc = sc / nsite
        mgg = sg / nsite
        mfm_acc += mf; ma_acc += maa; mc_acc += mcc; mg_acc += mgg
        mfm2 += mf*mf; ma2 += maa*maa; mc2 += mcc*mcc; mg2 += mgg*mgg
        mfm4 += mf**4; ma4 += maa**4; mc4 += mcc**4; mg4 += mgg**4
        abs_mfm += abs(mf); abs_ma += abs(maa); abs_mc += abs(mcc); abs_mg += abs(mgg)

        # Per-layer axial structure factor; sum the two layer spectra, avoiding A cancellation.
        for n in range(L):
            qy = 2.0 * math.pi * n / L
            amp2 = 0.0
            for l in range(2):
                re = 0.0; im = 0.0
                for x in range(L):
                    for y in range(L):
                        ang = qy * y
                        v = float(s[l, x, y, t])
                        re += v * math.cos(ang)
                        im += v * math.sin(ang)
                amp2 += re*re + im*im
            val = amp2 / nsite
            Sq[n] += val
            # normalized order amplitude squared ~ S/N
            a2 = amp2 / (nsite*nsite)
            q2[n] += a2
            q4[n] += a2*a2

    norm_bond = 2.0 * L * L * Mtau
    cx /= norm_bond; cy /= norm_bond; c2 /= norm_bond
    cdiag /= norm_bond; cx2 /= norm_bond
    cp /= (L * L * Mtau)
    temporal /= norm_bond
    invM = 1.0 / Mtau
    mfm_acc *= invM; ma_acc *= invM; mc_acc *= invM; mg_acc *= invM
    mfm2 *= invM; ma2 *= invM; mc2 *= invM; mg2 *= invM
    mfm4 *= invM; ma4 *= invM; mc4 *= invM; mg4 *= invM
    abs_mfm *= invM; abs_ma *= invM; abs_mc *= invM; abs_mg *= invM
    Sq *= invM; q2 *= invM; q4 *= invM

    # PIMC estimator for transverse magnetization, derived from d ln Z / d Gamma.
    if Gamma <= 1e-14 or Mtau == 1:
        mx = 0.0
    else:
        dtau = beta / Mtau
        a = dtau * Gamma
        sh = math.sinh(2.0*a)
        ch = math.cosh(2.0*a)
        mx = ch/sh - temporal/sh
        if mx > 1.0 + 1e-6:
            mx = 1.0
        if mx < -1.0 - 1e-6:
            mx = -1.0

    return (cx, cy, c2, cp, cdiag, cx2, temporal, mx,
            mfm_acc, ma_acc, mc_acc, mg_acc,
            mfm2, ma2, mc2, mg2,
            mfm4, ma4, mc4, mg4,
            abs_mfm, abs_ma, abs_mc, abs_mg, Sq, q2, q4)


@njit(parallel=True, cache=True)
def _run_pimc_replicas(params, L, beta, Mtau, seeds, therm_sweeps, meas_sweeps, measure_every):
    # params rows: Jx, Jy, J2, Jperp, Gamma
    P = params.shape[0]
    R = seeds.shape[0]
    Q = P * R
    nmeas = max(1, meas_sweeps // measure_every)
    S = np.empty((Q, 2, L, L, Mtau), dtype=np.int8)
    # 27 scalar accumulators + Sq mean per replica + time series primary order.
    scal = np.zeros((Q, 31), dtype=np.float64)
    sqacc = np.zeros((Q, L), dtype=np.float64)
    q2acc = np.zeros((Q, L), dtype=np.float64)
    q4acc = np.zeros((Q, L), dtype=np.float64)
    ts = np.zeros((Q, nmeas), dtype=np.float64)
    accept = np.zeros(Q, dtype=np.float64)

    for qidx in prange(Q):
        p = qidx // R
        r = qidx % R
        Jx, Jy, J2, Jp, G = params[p]
        rng = np.uint64(seeds[r] + np.uint64(104729 * (p + 1)) + np.uint64(1000003 * (r + 1)))
        init_code = r % 4
        rng = _initial_pattern(S, qidx, L, Mtau, init_code, rng)
        dtau = beta / Mtau if Mtau > 1 else beta
        Kx = dtau * Jx
        Ky = dtau * Jy
        K2 = -dtau * J2
        Kp = dtau * Jp
        if G > 1e-14 and Mtau > 1:
            a = dtau * G
            Kt = 0.5 * math.log(1.0 / math.tanh(a))
        else:
            Kt = 0.0

        total_attempt = 0
        total_accept = 0
        meas_idx = 0
        total_sweeps = therm_sweeps + meas_sweeps
        for sw in range(total_sweeps):
            # 32-color decomposition eliminates couplings inside a color:
            # layer(2) x x parity(2) x y mod 4(4) x tau parity(2).
            # For Mtau=1 only tau parity 0 is used.
            tcolors = 2 if Mtau > 1 else 1
            for lc in range(2):
                for xp in range(2):
                    for ym in range(4):
                        for tp in range(tcolors):
                            for x in range(xp, L, 2):
                                for y in range(ym, L, 4):
                                    for t in range(tp, Mtau, tcolors):
                                        s0 = float(S[qidx, lc, x, y, t])
                                        h = Kx * (S[qidx, lc, (x-1)%L, y, t] + S[qidx, lc, (x+1)%L, y, t])
                                        h += Ky * (S[qidx, lc, x, (y-1)%L, t] + S[qidx, lc, x, (y+1)%L, t])
                                        h += K2 * (S[qidx, lc, x, (y-2)%L, t] + S[qidx, lc, x, (y+2)%L, t])
                                        h += Kp * S[qidx, 1-lc, x, y, t]
                                        if Mtau > 1:
                                            h += Kt * (S[qidx, lc, x, y, (t-1)%Mtau] + S[qidx, lc, x, y, (t+1)%Mtau])
                                        dlog = -2.0 * s0 * h
                                        total_attempt += 1
                                        if dlog >= 0.0:
                                            S[qidx, lc, x, y, t] = -S[qidx, lc, x, y, t]
                                            total_accept += 1
                                        else:
                                            rng, u = _xorshift64star(rng)
                                            if u < math.exp(dlog):
                                                S[qidx, lc, x, y, t] = -S[qidx, lc, x, y, t]
                                                total_accept += 1
            # Full-worldline flips leave the temporal action invariant and strongly
            # improve ergodicity in the small-Gamma regime where K_tau is large.
            # They complement local flips, which still create/move imaginary-time domain walls.
            if Mtau > 1:
                for lc in range(2):
                    for x in range(L):
                        for y in range(L):
                            dlog_w = 0.0
                            for t in range(Mtau):
                                s0 = float(S[qidx, lc, x, y, t])
                                hsp = Kx * (S[qidx, lc, (x-1)%L, y, t] + S[qidx, lc, (x+1)%L, y, t])
                                hsp += Ky * (S[qidx, lc, x, (y-1)%L, t] + S[qidx, lc, x, (y+1)%L, t])
                                hsp += K2 * (S[qidx, lc, x, (y-2)%L, t] + S[qidx, lc, x, (y+2)%L, t])
                                hsp += Kp * S[qidx, 1-lc, x, y, t]
                                dlog_w += -2.0 * s0 * hsp
                            total_attempt += Mtau
                            if dlog_w >= 0.0:
                                for t in range(Mtau): S[qidx, lc, x, y, t] = -S[qidx, lc, x, y, t]
                                total_accept += Mtau
                            else:
                                rng, u = _xorshift64star(rng)
                                if u < math.exp(dlog_w):
                                    for t in range(Mtau): S[qidx, lc, x, y, t] = -S[qidx, lc, x, y, t]
                                    total_accept += Mtau
            if sw >= therm_sweeps and ((sw - therm_sweeps + 1) % measure_every == 0):
                m = _measure_replica(S[qidx], L, Mtau, beta, G)
                # scalar positions (explicit for Numba tuple typing)
                scal[qidx, 0] += m[0]; scal[qidx, 1] += m[1]; scal[qidx, 2] += m[2]; scal[qidx, 3] += m[3]
                scal[qidx, 4] += m[4]; scal[qidx, 5] += m[5]; scal[qidx, 6] += m[6]; scal[qidx, 7] += m[7]
                scal[qidx, 8] += m[8]; scal[qidx, 9] += m[9]; scal[qidx,10] += m[10]; scal[qidx,11] += m[11]
                scal[qidx,12] += m[12]; scal[qidx,13] += m[13]; scal[qidx,14] += m[14]; scal[qidx,15] += m[15]
                scal[qidx,16] += m[16]; scal[qidx,17] += m[17]; scal[qidx,18] += m[18]; scal[qidx,19] += m[19]
                scal[qidx,20] += m[20]; scal[qidx,21] += m[21]; scal[qidx,22] += m[22]; scal[qidx,23] += m[23]
                # energy estimators: diagonal and total
                diag_e_per_site = -Jx*m[0] - Jy*m[1] + J2*m[2] - 0.5*Jp*m[3]
                total_e_per_site = diag_e_per_site - G*m[7]
                scal[qidx, 24] += diag_e_per_site
                scal[qidx, 25] += total_e_per_site
                # susceptibility-like values for four templates use per measurement m moments.
                scal[qidx, 26] += m[12]
                scal[qidx, 27] += m[13]
                scal[qidx, 28] += m[14]
                scal[qidx, 29] += m[15]
                scal[qidx, 30] += max(m[12], m[13], m[14], m[15])
                sqacc[qidx] += m[24]
                q2acc[qidx] += m[25]
                q4acc[qidx] += m[26]
                ts[qidx, meas_idx] = math.sqrt(max(0.0, scal[qidx, 30] / (meas_idx + 1)))
                meas_idx += 1
        if meas_idx > 0:
            scal[qidx] /= meas_idx
            sqacc[qidx] /= meas_idx
            q2acc[qidx] /= meas_idx
            q4acc[qidx] /= meas_idx
        accept[qidx] = total_accept / max(1, total_attempt)
    return scal, sqacc, q2acc, q4acc, ts, accept


def run_pimc_batch(param_rows, L, T, Mtau, seeds, therm, meas, every, num_threads=4):
    if not NUMBA_OK:
        raise RuntimeError("Numba is required for production PIMC")
    set_num_threads(max(1, num_threads))
    beta = 1.0 / T
    pars = np.asarray([[r["Jx"], r["Jy"], r["J2"], r["Jperp"], r["Gamma"]] for r in param_rows], dtype=np.float64)
    seeds_arr = np.asarray(seeds, dtype=np.uint64)
    scal, sq, q2, q4, ts, acc = _run_pimc_replicas(pars, int(L), float(beta), int(Mtau), seeds_arr,
                                                     int(therm), int(meas), int(every))
    out = []
    R = len(seeds)
    for p, prow in enumerate(param_rows):
        for r, seed in enumerate(seeds):
            qidx = p * R + r
            s = scal[qidx]
            Sq = sq[qidx]
            nstar = int(np.argmax(Sq))
            qstar = 2.0 * math.pi * nstar / L
            # fold q into [0, pi]
            if qstar > math.pi:
                qstar = 2.0 * math.pi - qstar
            Smax = float(Sq[nstar])
            N = 2 * L * L
            ordfrac = Smax / N
            # second-moment xi along y, neighbor momentum around qstar
            nb = (nstar + 1) % L
            denom = max(float(Sq[nb]), 1e-12)
            ratio = max(Smax / denom - 1.0, 0.0)
            xi = 0.5 / max(math.sin(math.pi / L), 1e-12) * math.sqrt(ratio)
            # Binder for canonical order parameters.
            binders = []
            for m2idx, m4idx in [(12,16),(13,17),(14,18),(15,19)]:
                m2 = max(s[m2idx], 1e-14)
                m4 = s[m4idx]
                binders.append(1.0 - m4 / (3.0 * m2 * m2))
            tau = autocorr_time(ts[qidx])
            row = dict(prow)
            row.update({
                "L": int(L), "T": float(T), "Mtau": int(Mtau), "seed": int(seed),
                "init_code": int(r % 4), "therm_sweeps": int(therm), "meas_sweeps": int(meas),
                "measure_every": int(every), "acceptance": float(acc[qidx]), "tau_int": tau,
                "Cxzz": float(s[0]), "Cyzz": float(s[1]), "C2zz": float(s[2]),
                "Cperpzz": float(s[3]), "Cdiagzz": float(s[4]), "Cx2zz": float(s[5]),
                "Ctau": float(s[6]), "mx": float(s[7]),
                "mFM": float(s[8]), "mA": float(s[9]), "mC": float(s[10]), "mG": float(s[11]),
                "mFM2": float(s[12]), "mA2": float(s[13]), "mC2": float(s[14]), "mG2": float(s[15]),
                "mFM4": float(s[16]), "mA4": float(s[17]), "mC4": float(s[18]), "mG4": float(s[19]),
                "abs_mFM": float(s[20]), "abs_mA": float(s[21]), "abs_mC": float(s[22]), "abs_mG": float(s[23]),
                "Ediag_per_spin": float(s[24]), "E_per_spin": float(s[25]),
                "qstar_y": qstar, "qstar_index": nstar, "Smax": Smax, "Smax_over_N": ordfrac,
                "xi_y": xi, "xi_over_L": xi / L,
                "U4_FM": binders[0], "U4_A": binders[1], "U4_C": binders[2], "U4_G": binders[3],
                "qmc_qc": "PASS" if (tau < max(2.0, 0.20*len(ts[qidx])) and 0.001 < float(acc[qidx]) < 0.999) else "CHECK",
                "Sq_y_json": json.dumps([float(x) for x in Sq]),
                "q2_y_json": json.dumps([float(x) for x in q2[qidx]]),
                "q4_y_json": json.dumps([float(x) for x in q4[qidx]]),
            })
            out.append(row)
    return out

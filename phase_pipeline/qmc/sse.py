"""Stochastic series expansion (SSE) for transverse-field Ising models with arbitrary Ising bonds.

The Hamiltonian H = -sum_b J_b s^z_i s^z_j - Gamma sum_i s^x_i is written, following
Sandvik, PRE 68, 056701 (2003), as

    H = -sum_b H_b - sum_i (H_{i,c} + H_{i,f}) + C,
    H_b     = |J_b| + J_b s^z_i s^z_j   (2|J_b| on a satisfied bond, 0 otherwise),
    H_{i,c} = Gamma                     (constant site operator),
    H_{i,f} = Gamma s^x_i               (spin flip),
    C       = sum_b |J_b| + N Gamma,

so that E = -<n>/beta + C. Diagonal updates insert and remove H_b and H_{i,c}; the cluster
update builds Swendsen-Wang-type clusters that pass through all four legs of a bond operator
and end on site operators, and flips each with probability 1/2 (a site operator whose two legs
end up in clusters with different flips changes between H_{i,c} and H_{i,f}). Every weight is
positive for either sign of J_b, so frustration (J2 > 0) costs efficiency, not correctness.
There is no Trotter error.

The engine takes a generic bond list. Bonds are grouped in classes with a common |J| (for the
bilayer: x, y, next-nearest y, rung); pair classes, templates and the per-layer axial
structure factor are measured on the propagated state at imaginary time 0.

A chain runs through a schedule of Gamma values at fixed beta, carrying its configuration from
one value to the next; ordering the schedule from large to small Gamma anneals the chain out of
the paramagnet.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange, set_num_threads

from .engine import _xorshift64star as _rand

# Columns of the per-step scalar output (before the per-class and per-template blocks).
S_N, S_N2, S_NFLIP = 0, 1, 2
N_HEAD = 3
# Time-series columns (binned): n, n_flip, m_FM^2, m_A^2, and the layer-averaged |a_l(q)|^2
# at q = 0, pi/2, pi/3 (-1 if Ly does not allow it) and its maximum over q.
TS_COLS = ["n", "n_flip", "mFM2", "mA2", "Sl_q0", "Sl_qpi2", "Sl_qpi3", "Sl_max"]
N_TS = len(TS_COLS)


# ---------------------------------------------------------------------------------------------
# Lattices
# ---------------------------------------------------------------------------------------------

BOND_CLASSES = ["x", "y", "y2", "perp"]
PAIR_CLASSES = ["Cx", "Cy", "C2", "Cperp", "Cdiag", "Cx2"]
TEMPLATES = ["FM", "A", "C", "G"]


def bilayer_lattice(Lx, Ly, nlayers=2):
    """Periodic Lx x Ly layers; site index l*Lx*Ly + x*Ly + y.

    Bond classes 0..3 = x, y, next-nearest y, rung (rungs only for two layers). Pair classes
    0..5 = x, y, y2, rung, diagonal (x+1, y+1), second neighbor along x; the last two are not
    bonds. Small periodic lengths repeat bonds (e.g. Ly = 4 gives each y2 pair twice); the
    Hamiltonian then contains the pair twice, and exact diagonalization must use the same list.
    """
    def s(l, x, y):
        return l * Lx * Ly + (x % Lx) * Ly + (y % Ly)

    bonds = []
    pairs = []
    for l in range(nlayers):
        for x in range(Lx):
            for y in range(Ly):
                if Lx > 1:
                    bonds.append((s(l, x, y), s(l, x + 1, y), 0))
                    pairs.append((s(l, x, y), s(l, x + 1, y), 0))
                    pairs.append((s(l, x, y), s(l, x + 1, y + 1), 4))
                bonds.append((s(l, x, y), s(l, x, y + 1), 1))
                bonds.append((s(l, x, y), s(l, x, y + 2), 2))
                pairs.append((s(l, x, y), s(l, x, y + 1), 1))
                pairs.append((s(l, x, y), s(l, x, y + 2), 2))
                pairs.append((s(l, x, y), s(l, x + 2, y), 5))
    if nlayers == 2:
        for x in range(Lx):
            for y in range(Ly):
                bonds.append((s(0, x, y), s(1, x, y), 3))
                pairs.append((s(0, x, y), s(1, x, y), 3))
    # measurement-only pairs that fold onto one site on small clusters are dropped
    pairs = [p for p in pairs if p[0] != p[1]]
    N = nlayers * Lx * Ly
    layer = np.array([i // (Lx * Ly) for i in range(N)], dtype=np.int64)
    ycoord = np.array([i % Ly for i in range(N)], dtype=np.int64)
    tmpl = np.empty((4, N), dtype=np.float64)
    tmpl[0] = 1.0
    tmpl[1] = np.where(layer == 0, 1.0, -1.0)
    tmpl[2] = np.where(ycoord % 2 == 0, 1.0, -1.0)
    tmpl[3] = tmpl[1] * tmpl[2]
    b = np.array(bonds, dtype=np.int64)
    pr = np.array(pairs, dtype=np.int64)
    if np.any(b[:, 0] == b[:, 1]) or np.any(pr[:, 0] == pr[:, 1]):
        raise ValueError(f"Lx={Lx}, Ly={Ly} maps a bond or pair onto a single site")
    return {"N": N, "Lx": Lx, "Ly": Ly, "nlayers": nlayers,
            "bi": b[:, 0], "bj": b[:, 1], "bcls": b[:, 2], "n_bcls": 4,
            "pi": pr[:, 0], "pj": pr[:, 1], "pcls": pr[:, 2], "n_pcls": 6,
            "tmpl": tmpl, "layer": layer, "ycoord": ycoord}


def class_couplings(Jx, Jy, J2, Jperp):
    """J_b per bond class in the convention H = -J_b s s (J2 enters as -J2)."""
    return np.array([Jx, Jy, -J2, Jperp], dtype=np.float64)


def initial_spins(lat, code, rng):
    """Initial spin configuration. Codes: 0 random, 1 FM, 2 A (antiparallel layers),
    3 <2> parallel layers, 4 <2> antiparallel, 5 <3> parallel, 6 <3> antiparallel."""
    layer, y = lat["layer"], lat["ycoord"]
    sgn = np.where(layer == 0, 1, -1)
    if code == 0:
        s = rng.choice([-1, 1], size=lat["N"])
    elif code == 1:
        s = np.ones(lat["N"], dtype=int)
    elif code == 2:
        s = sgn
    elif code in (3, 4):
        s = np.where(y % 4 < 2, 1, -1) * (sgn if code == 4 else 1)
    elif code in (5, 6):
        s = np.where(y % 6 < 3, 1, -1) * (sgn if code == 6 else 1)
    else:
        raise ValueError(code)
    return s.astype(np.int8)


# ---------------------------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------------------------

@njit(cache=True)
def _class_cumulative(Gamma, N, Jc, cls_count):
    ncls = Jc.shape[0] + 1
    cum = np.empty(ncls, dtype=np.float64)
    cum[0] = N * Gamma
    for c in range(1, ncls):
        cum[c] = cum[c - 1] + cls_count[c - 1] * 2.0 * abs(Jc[c - 1])
    return cum


@njit(cache=True)
def _diagonal_update(spins, optype, opidx, n, beta, N, bi, bj, bsign, cum, cls_first, cls_count, rng):
    M = optype.shape[0]
    Wtot = cum[cum.shape[0] - 1]
    ncls = cum.shape[0]
    for p in range(M):
        t = optype[p]
        if t == 0:
            rng, u = _rand(rng)
            if u * (M - n) < beta * Wtot:
                rng, u = _rand(rng)
                x = u * Wtot
                c = 0
                while c < ncls - 1 and x >= cum[c]:
                    c += 1
                rng, u = _rand(rng)
                if c == 0:
                    k = int(u * N)
                    if k >= N:
                        k = N - 1
                    optype[p] = 1
                    opidx[p] = k
                    n += 1
                else:
                    cnt = cls_count[c - 1]
                    k = int(u * cnt)
                    if k >= cnt:
                        k = cnt - 1
                    b = cls_first[c - 1] + k
                    if bsign[c - 1] * spins[bi[b]] * spins[bj[b]] > 0:
                        optype[p] = 3
                        opidx[p] = b
                        n += 1
        elif t == 2:
            spins[opidx[p]] = -spins[opidx[p]]
        else:
            rng, u = _rand(rng)
            if u * beta * Wtot < (M - n + 1):
                optype[p] = 0
                n -= 1
    return n, rng


@njit(cache=True)
def _cluster_update(spins, optype, opidx, N, bi, bj, vlink, vis, first, last, stack, rng):
    M = optype.shape[0]
    for i in range(N):
        first[i] = -1
        last[i] = -1
    for p in range(M):
        t = optype[p]
        if t == 0:
            continue
        v0 = 4 * p
        if t == 3:
            b = opidx[p]
            i = bi[b]
            j = bj[b]
            li = last[i]
            if li >= 0:
                vlink[li] = v0
                vlink[v0] = li
            else:
                first[i] = v0
            last[i] = v0 + 2
            lj = last[j]
            if lj >= 0:
                vlink[lj] = v0 + 1
                vlink[v0 + 1] = lj
            else:
                first[j] = v0 + 1
            last[j] = v0 + 3
        else:
            i = opidx[p]
            li = last[i]
            if li >= 0:
                vlink[li] = v0
                vlink[v0] = li
            else:
                first[i] = v0
            last[i] = v0 + 2
    for i in range(N):
        f = first[i]
        if f >= 0:
            l = last[i]
            vlink[f] = l
            vlink[l] = f

    for v in range(4 * M):
        vis[v] = 0
    for p in range(M):
        t = optype[p]
        if t == 0:
            continue
        for leg in range(4):
            if t != 3 and (leg == 1 or leg == 3):
                continue
            v0 = 4 * p + leg
            if vis[v0] != 0:
                continue
            rng, u = _rand(rng)
            flag = 2 if u < 0.5 else 1
            stack[0] = v0
            top = 1
            while top > 0:
                top -= 1
                v = stack[top]
                if vis[v] != 0:
                    continue
                q = v >> 2
                if optype[q] == 3:
                    base = 4 * q
                    for k in range(4):
                        w = base + k
                        if vis[w] == 0:
                            vis[w] = flag
                            wl = vlink[w]
                            if vis[wl] == 0:
                                stack[top] = wl
                                top += 1
                else:
                    vis[v] = flag
                    wl = vlink[v]
                    if vis[wl] == 0:
                        stack[top] = wl
                        top += 1

    for p in range(M):
        t = optype[p]
        if t == 1 or t == 2:
            if (vis[4 * p] == 2) != (vis[4 * p + 2] == 2):
                optype[p] = 3 - t
    for i in range(N):
        f = first[i]
        if f >= 0:
            if vis[f] == 2:
                spins[i] = -spins[i]
        else:
            rng, u = _rand(rng)
            if u < 0.5:
                spins[i] = -spins[i]
    return rng


@njit(cache=True)
def _measure(spins, N, pi, pj, pcls, n_pcls, pcount, tmpl, layer, ycoord, nlay, LxLy, Ly,
             cosq, sinq, pc_out, tm_out, sl2, sl4, ss2, ss4):
    """Equal-time observables of the state at imaginary time 0 (outputs are overwritten)."""
    for c in range(n_pcls):
        pc_out[c] = 0.0
    for k in range(pi.shape[0]):
        pc_out[pcls[k]] += spins[pi[k]] * spins[pj[k]]
    for c in range(n_pcls):
        if pcount[c] > 0:
            pc_out[c] /= pcount[c]
    for t in range(tmpl.shape[0]):
        m = 0.0
        for i in range(N):
            m += tmpl[t, i] * spins[i]
        tm_out[t] = m / N
    col = np.zeros((nlay, Ly), dtype=np.float64)
    for i in range(N):
        col[layer[i], ycoord[i]] += spins[i]
    for n in range(Ly):
        sr = 0.0
        si = 0.0
        a2l = 0.0
        a4l = 0.0
        for l in range(nlay):
            re = 0.0
            im = 0.0
            for y in range(Ly):
                re += col[l, y] * cosq[n, y]
                im += col[l, y] * sinq[n, y]
            re /= LxLy
            im /= LxLy
            a2 = re * re + im * im
            a2l += a2
            a4l += a2 * a2
            sr += re
            si += im
        sl2[n] = a2l / nlay
        sl4[n] = a4l / nlay
        sr /= nlay
        si /= nlay
        s2 = sr * sr + si * si
        ss2[n] = s2
        ss4[n] = s2 * s2


@njit(cache=True)
def _grow(optype, opidx, Mnew):
    M = optype.shape[0]
    ot = np.zeros(Mnew, dtype=np.int8)
    oi = np.zeros(Mnew, dtype=np.int32)
    # spread the existing operators over the longer string, keeping their order
    for p in range(M):
        q = (p * Mnew) // M
        ot[q] = optype[p]
        oi[q] = opidx[p]
    return ot, oi


@njit(cache=True)
def _run_chain(Jc, Gammas, beta, N, bi, bj, bcls_first, bcls_count,
               pi, pj, pcls, n_pcls, pcount, tmpl, layer, ycoord, nlay, LxLy, Ly, cosq, sinq,
               iq0, iq2, iq3, spins0, seed, therm_steps, meas_steps, bin_size,
               scal, spec, bins, final_spins, diag, prefill):
    # therm_steps[k], meas_steps[k]: sweeps at the k-th field; meas_steps[k] = 0 means the
    # step only anneals (no measurement, its output rows stay zero).
    # prefill: diagonal-only sweeps at the first field before any cluster update. The string
    # starts empty, and a cluster update on a nearly empty string flips every operator-free
    # spin with probability 1/2, which would randomize the initial configuration; filling
    # the string first (the spins cannot change during diagonal updates) keeps a template
    # start intact. It is part of thermalization and leaves the sampled distribution unchanged.
    N_T = tmpl.shape[0]
    ncb = Jc.shape[0]
    bsign = np.empty(ncb, dtype=np.float64)
    for c in range(ncb):
        bsign[c] = 1.0 if Jc[c] >= 0 else -1.0
    spins = spins0.copy()
    rng = np.uint64(seed)
    for _ in range(10):
        rng, u = _rand(rng)
    M = 64
    optype = np.zeros(M, dtype=np.int8)
    opidx = np.zeros(M, dtype=np.int32)
    n = 0
    first = np.empty(N, dtype=np.int32)
    last = np.empty(N, dtype=np.int32)
    vlink = np.empty(4 * M, dtype=np.int32)
    vis = np.empty(4 * M, dtype=np.int8)
    stack = np.empty(4 * M + 1, dtype=np.int32)
    pc = np.empty(n_pcls, dtype=np.float64)
    tm = np.empty(N_T, dtype=np.float64)
    sl2 = np.empty(Ly, dtype=np.float64)
    sl4 = np.empty(Ly, dtype=np.float64)
    ss2 = np.empty(Ly, dtype=np.float64)
    ss4 = np.empty(Ly, dtype=np.float64)
    nbins = bins.shape[1]

    cum0 = _class_cumulative(Gammas[0], N, Jc, bcls_count)
    for _ in range(prefill):
        n, rng = _diagonal_update(spins, optype, opidx, n, beta, N, bi, bj, bsign,
                                  cum0, bcls_first, bcls_count, rng)
        if n > (3 * M) // 4:
            Mnew = n + n // 2 + 64
            optype, opidx = _grow(optype, opidx, Mnew)
            M = Mnew
            vlink = np.empty(4 * M, dtype=np.int32)
            vis = np.empty(4 * M, dtype=np.int8)
            stack = np.empty(4 * M + 1, dtype=np.int32)

    for k in range(Gammas.shape[0]):
        G = Gammas[k]
        n_therm = therm_steps[k]
        n_meas = meas_steps[k]
        cum = _class_cumulative(G, N, Jc, bcls_count)
        for sw in range(n_therm + n_meas):
            n, rng = _diagonal_update(spins, optype, opidx, n, beta, N, bi, bj, bsign,
                                      cum, bcls_first, bcls_count, rng)
            if sw < n_therm and n > (3 * M) // 4:
                Mnew = n + n // 2 + 64
                optype, opidx = _grow(optype, opidx, Mnew)
                M = Mnew
                vlink = np.empty(4 * M, dtype=np.int32)
                vis = np.empty(4 * M, dtype=np.int8)
                stack = np.empty(4 * M + 1, dtype=np.int32)
            rng = _cluster_update(spins, optype, opidx, N, bi, bj, vlink, vis, first, last, stack, rng)
            if sw < n_therm:
                continue
            ms = sw - n_therm
            nflip = 0
            ncls = np.zeros(ncb, dtype=np.float64)
            for p in range(M):
                if optype[p] == 2:
                    nflip += 1
                elif optype[p] == 3:
                    b = opidx[p]
                    for c in range(ncb):
                        if b >= bcls_first[c] and b < bcls_first[c] + bcls_count[c]:
                            ncls[c] += 1.0
                            break
            _measure(spins, N, pi, pj, pcls, n_pcls, pcount, tmpl, layer, ycoord, nlay, LxLy, Ly,
                     cosq, sinq, pc, tm, sl2, sl4, ss2, ss4)
            scal[k, S_N] += n
            scal[k, S_N2] += float(n) * float(n)
            scal[k, S_NFLIP] += nflip
            o = N_HEAD
            for c in range(ncb):
                scal[k, o + c] += ncls[c]
            o += ncb
            for c in range(n_pcls):
                scal[k, o + c] += pc[c]
            o += n_pcls
            for t in range(N_T):
                m = tm[t]
                scal[k, o + 4 * t + 0] += m
                scal[k, o + 4 * t + 1] += m * m
                scal[k, o + 4 * t + 2] += m * m * m * m
                scal[k, o + 4 * t + 3] += abs(m)
            smax = 0.0
            for q in range(Ly):
                spec[k, q, 0] += sl2[q]
                spec[k, q, 1] += sl4[q]
                spec[k, q, 2] += ss2[q]
                spec[k, q, 3] += ss4[q]
                if sl2[q] > smax:
                    smax = sl2[q]
            bi_ = ms // bin_size
            if bi_ < nbins:
                bins[k, bi_, 0] += n
                bins[k, bi_, 1] += nflip
                bins[k, bi_, 2] += tm[0] * tm[0]
                bins[k, bi_, 3] += tm[1] * tm[1]
                bins[k, bi_, 4] += sl2[iq0]
                bins[k, bi_, 5] += sl2[iq2] if iq2 >= 0 else -1.0
                bins[k, bi_, 6] += sl2[iq3] if iq3 >= 0 else -1.0
                bins[k, bi_, 7] += smax
        inv = 1.0 / n_meas if n_meas > 0 else 0.0
        for c in range(scal.shape[1]):
            scal[k, c] *= inv
        for q in range(Ly):
            for c in range(4):
                spec[k, q, c] *= inv
        for b_ in range(nbins):
            for c in range(N_TS):
                bins[k, b_, c] /= bin_size
        for i in range(N):
            final_spins[k, i] = spins[i]
        diag[k, 0] = M
        diag[k, 1] = n
    return 0


@njit(parallel=True, cache=True)
def _run_chains(Jc_all, Gammas_all, beta, N, bi, bj, bcls_first, bcls_count,
                pi, pj, pcls, n_pcls, pcount, tmpl, layer, ycoord, nlay, LxLy, Ly, cosq, sinq,
                iq0, iq2, iq3, spins0_all, seeds, therm_steps, meas_steps, bin_size,
                scal, spec, bins, final_spins, diag, prefill):
    for c in prange(Jc_all.shape[0]):
        _run_chain(Jc_all[c], Gammas_all[c], beta, N, bi, bj, bcls_first, bcls_count,
                   pi, pj, pcls, n_pcls, pcount, tmpl, layer, ycoord, nlay, LxLy, Ly, cosq, sinq,
                   iq0, iq2, iq3, spins0_all[c], seeds[c], therm_steps, meas_steps, bin_size,
                   scal[c], spec[c], bins[c], final_spins[c], diag[c], prefill)


# ---------------------------------------------------------------------------------------------
# Python driver
# ---------------------------------------------------------------------------------------------

def run_sse_chains(lat, couplings, gamma_schedules, beta, init_spins, seeds,
                   n_therm, n_meas, bin_size=16, threads=6, prefill=200):
    """Run independent SSE chains on one lattice geometry.

    couplings: (C, n_bcls) array of J per bond class for each chain (class_couplings()).
    gamma_schedules: (C, K) array; each chain visits its K fields in order.
    init_spins: (C, N) int8. Returns a list (per chain) of lists (per field) of result dicts.
    n_therm, n_meas: sweeps per field, either one int for every field or one value per field
    (length K); a field with n_meas = 0 is only annealed through, and its result row is empty
    (all estimators zero or nan).
    prefill: diagonal-only sweeps at the first field before the first cluster update, so that
    init_spins survive the start (see _run_chain); 0 reproduces the earlier behavior.
    """
    set_num_threads(max(1, threads))
    order = np.argsort(lat["bcls"], kind="stable")
    bi = lat["bi"][order].astype(np.int32)
    bj = lat["bj"][order].astype(np.int32)
    bcls = lat["bcls"][order]
    nb = lat["n_bcls"]
    bcls_count = np.array([np.sum(bcls == c) for c in range(nb)], dtype=np.int64)
    bcls_first = np.concatenate([[0], np.cumsum(bcls_count)[:-1]]).astype(np.int64)
    pcls = lat["pcls"].astype(np.int64)
    n_pcls = lat["n_pcls"]
    pcount = np.array([np.sum(pcls == c) for c in range(n_pcls)], dtype=np.float64)
    Ly, N = lat["Ly"], lat["N"]
    LxLy = float(N // lat["nlayers"])
    qs = 2.0 * np.pi * np.arange(Ly) / Ly
    cosq = np.cos(np.outer(qs, np.arange(Ly)))
    sinq = np.sin(np.outer(qs, np.arange(Ly)))
    iq2 = Ly // 4 if Ly % 4 == 0 else -1
    iq3 = Ly // 6 if Ly % 6 == 0 else -1
    Jc_all = np.ascontiguousarray(couplings, dtype=np.float64)
    G_all = np.ascontiguousarray(gamma_schedules, dtype=np.float64)
    C, K = G_all.shape
    therm_steps = np.broadcast_to(np.asarray(n_therm, dtype=np.int64), (K,)).copy()
    meas_steps = np.broadcast_to(np.asarray(n_meas, dtype=np.int64), (K,)).copy()
    N_T = lat["tmpl"].shape[0]
    ncol = N_HEAD + nb + n_pcls + 4 * N_T
    nbins = max(1, int(meas_steps.max()) // bin_size)
    scal = np.zeros((C, K, ncol))
    spec = np.zeros((C, K, Ly, 4))
    bins = np.zeros((C, K, nbins, N_TS))
    final = np.zeros((C, K, N), dtype=np.int8)
    diag = np.zeros((C, K, 2))
    _run_chains(Jc_all, G_all, float(beta), N, bi, bj, bcls_first, bcls_count,
                lat["pi"].astype(np.int32), lat["pj"].astype(np.int32), pcls, n_pcls, pcount,
                lat["tmpl"], lat["layer"], lat["ycoord"], lat["nlayers"], LxLy, Ly, cosq, sinq,
                0, iq2, iq3, np.ascontiguousarray(init_spins, dtype=np.int8),
                np.asarray(seeds, dtype=np.uint64), therm_steps, meas_steps, int(bin_size),
                scal, spec, bins, final, diag, int(prefill))

    out = []
    for c in range(C):
        Jc = Jc_all[c]
        const_b = float(np.sum(bcls_count * np.abs(Jc)))
        rows = []
        for k in range(K):
            G = G_all[c, k]
            s = scal[c, k]
            nmean = s[S_N]
            r = {"Gamma": float(G), "beta": float(beta), "N": N,
                 "n_therm": int(therm_steps[k]), "n_meas": int(meas_steps[k]),
                 "E_per_spin": (-nmean / beta + const_b + N * G) / N,
                 "C_per_spin": (s[S_N2] - nmean * nmean - nmean) / N,
                 "mx": s[S_NFLIP] / (beta * G * N) if G > 0 else float("nan"),
                 "M_cutoff": int(diag[c, k, 0]), "n_ops": int(diag[c, k, 1])}
            o = N_HEAD
            for ci, name in enumerate(BOND_CLASSES[:nb]):
                if abs(Jc[ci]) > 0 and bcls_count[ci] > 0:
                    # <H_b> = <n_b>/beta per bond; H_b = |J| + J s s
                    r[f"bond_{name}"] = (s[o + ci] / (beta * bcls_count[ci]) - abs(Jc[ci])) / Jc[ci]
                else:
                    r[f"bond_{name}"] = float("nan")
            o += nb
            for ci, name in enumerate(PAIR_CLASSES[:n_pcls]):
                r[name] = float(s[o + ci])
            o += n_pcls
            for t, name in enumerate(TEMPLATES[:N_T]):
                r[f"m{name}"], r[f"m{name}2"], r[f"m{name}4"], r[f"abs_m{name}"] = (
                    float(v) for v in s[o + 4 * t: o + 4 * t + 4])
            r["Sl2"] = spec[c, k, :, 0].copy()
            r["Sl4"] = spec[c, k, :, 1].copy()
            r["Ss2"] = spec[c, k, :, 2].copy()
            r["Ss4"] = spec[c, k, :, 3].copy()
            r["bins"] = bins[c, k, :int(meas_steps[k]) // bin_size].copy()
            r["final_spins"] = final[c, k].copy()
            if meas_steps[k] == 0:
                # annealing-only step: nothing was measured, so no estimator is defined
                keep = {"Gamma", "beta", "N", "n_therm", "n_meas", "M_cutoff", "n_ops", "bins", "final_spins"}
                for key in list(r):
                    if key in keep:
                        continue
                    if isinstance(r[key], np.ndarray):
                        r[key] = np.full_like(r[key], np.nan, dtype=np.float64)
                    else:
                        r[key] = float("nan")
            rows.append(r)
        out.append(rows)
    return out

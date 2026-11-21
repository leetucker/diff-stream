"""Unit influence scores of the continuous-treatment (dose) DiD, one batch at a time."""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def dose_scores(
    gidx,
    y,
    psi,
    dpsi,
    ctl,
    gi,
    ti,
    bases,
    controls,
    nc,
    means,
    control_mean,
    sizes,
    beta,
    acrt,
    ainv,
    dbar,
    cbar,
    local,
    width,
):
    """Each unit's scores, in sum scale (the covariance is their sum of outer products).

    The scores are, for the M cells: the binarized ATT (M), the average causal
    response on the treated (M), each cell's spline coefficients (M blocks of K) and
    each cell's control-group mean change (M). ``local[k, c]`` is the output column of
    score c for a unit of cohort k: either c itself (one layout for all units, with
    ``width`` the number of scores) or the column in a layout of the scores that can
    be nonzero for that cohort (``width`` the largest such number). ``psi`` and
    ``dpsi`` are the units' basis values and derivatives (zero if never treated);
    a unit's ACRT is ``dpsi @ beta - ctl * control_mean``, and ``cbar[k]`` is the
    mean of ``ctl`` over cohort k (both zero for a spline; with a discrete dose,
    1 / d_1 at the lowest dose level d_1).
    """
    B = y.shape[0]
    M = len(gi)
    K = psi.shape[1]
    out = np.zeros((B, width))
    first_beta = 2 * M
    first_mean = 2 * M + M * K
    for i in prange(B):
        k = gidx[i]
        for j in range(M):
            change = y[i, ti[j]] - y[i, bases[j]]
            if k == gi[j]:
                out[i, local[k, j]] = (change - means[k, ti[j]] + means[k, bases[j]]) / sizes[k]
                fit = 0.0
                slope = -ctl[i] * control_mean[j]
                for a in range(K):
                    fit += psi[i, a] * beta[j, a]
                    slope += dpsi[i, a] * beta[j, a]
                e = change - fit
                column = first_beta + j * K
                projected = 0.0
                for a in range(K):
                    u = 0.0
                    for b in range(K):
                        u += ainv[k, a, b] * psi[i, b]
                    u *= e
                    out[i, local[k, column + a]] = u
                    projected += dbar[k, a] * u
                out[i, local[k, M + j]] = (slope - acrt[j]) / sizes[k] + projected
            elif controls[j, k]:
                s = (change - control_mean[j]) / nc[j]
                out[i, local[k, j]] = -s
                out[i, local[k, first_mean + j]] = s
                if cbar[gi[j]] != 0.0:
                    out[i, local[k, M + j]] = -cbar[gi[j]] * s
    return out


@njit(cache=True, parallel=True)
def cell_comparison_sums(gidx, y, X, levels, w, gi, ti, bases, period, controls, n, sx, sy, xx, xy):
    """Weighted sums of each cell's comparison regression, by fixed-effect level.

    A unit enters cell j if its group is one of the cell's comparisons and its outcome
    is observed in the period compared and the base period, with weight
    ``w[i, ti[j]]`` (its weight in the period compared) and its covariates and level
    at the cell's covariate period ``period[j]``. ``n[j, l]`` sums the weights,
    ``sx[j, l]`` the weighted covariates and ``sy[j, l]`` the weighted outcome
    changes; ``xx[j]`` and ``xy[j]`` the weighted products. Cells are summed in
    parallel, so no two threads write the same entries.
    """
    B = y.shape[0]
    M = len(gi)
    P = X.shape[2]
    for j in prange(M):
        t, b, p = ti[j], bases[j], period[j]
        for i in range(B):
            if not controls[j, gidx[i]]:
                continue
            yt, yb = y[i, t], y[i, b]
            if not (np.isfinite(yt) and np.isfinite(yb)):
                continue
            om = w[i, t]
            lv = levels[i, p]
            dy = yt - yb
            n[j, lv] += om
            sy[j, lv] += om * dy
            for a in range(P):
                xa = X[i, p, a]
                sx[j, lv, a] += om * xa
                xy[j, a] += om * xa * dy
                for c in range(P):
                    xx[j, a, c] += om * xa * X[i, p, c]


@njit(cache=True, parallel=True)
def kept_units(
    gidx, treated, y, X, levels, gi, ti, bases, period, present, probe, probe_x, tolerance
):
    """Whether each treated unit's adjustment is identified in every cell of its cohort
    in which the unit is observed (other units are always kept).

    It is identified when each of its fixed-effect levels has comparison units in the
    cell (``present[j, f, level]``) and its combination of levels is estimable from
    them: ``probe`` (by term and level) and ``probe_x`` (by covariate) are a generic
    vector of the null space of the cell's comparison design, so the unit's design z
    is estimable when z'probe is zero (within ``tolerance``). With one term, or no
    fixed effect, every combination of present levels is estimable and ``probe`` is
    zero.
    """
    B = y.shape[0]
    F = levels.shape[2]
    P = X.shape[2]
    out = np.ones(B, dtype=np.bool_)
    for i in prange(B):
        if not treated[i]:
            continue
        for j in range(len(gi)):
            if gi[j] != gidx[i]:
                continue
            if not (np.isfinite(y[i, ti[j]]) and np.isfinite(y[i, bases[j]])):
                continue
            p = period[j]
            value = 0.0
            for f in range(F):
                lv = levels[i, p, f]
                if not present[j, f, lv]:
                    out[i] = False
                value += probe[j, f, lv]
            for c in range(P):
                value += X[i, p, c] * probe_x[j, c]
            if abs(value) > tolerance:
                out[i] = False
            if not out[i]:
                break
    return out


@njit(cache=True, parallel=True)
def cell_treated_sums(
    gidx,
    kept,
    y,
    X,
    levels,
    w,
    psi,
    dpsi,
    gi,
    ti,
    bases,
    period,
    gamma,
    alpha,
    A,
    S,
    total,
    size,
    slope,
    level_basis,
    level_count,
    basis_x,
    treated_x,
):
    """Weighted moments of each cell's treated units, on their adjusted changes.

    A kept unit of the cell's cohort that is observed in both of its periods has
    adjusted change ``a = dY - X'gamma - sum_f alpha[f, level_f]`` and weight
    ``w[i, ti[j]]``.
    The sums are those of the dose regression (``A``, ``S``), the binarized ATT
    (``total``, ``size``), the average slope of the basis (``slope``), and what the
    comparison units' influence needs (by term and level, and with the covariates).
    """
    B = y.shape[0]
    M = len(gi)
    K = psi.shape[1]
    P = X.shape[2]
    F = levels.shape[2]
    for j in prange(M):
        g, t, b, p = gi[j], ti[j], bases[j], period[j]
        for i in range(B):
            if gidx[i] != g or not kept[i]:
                continue
            yt, yb = y[i, t], y[i, b]
            if not (np.isfinite(yt) and np.isfinite(yb)):
                continue
            om = w[i, t]
            a = yt - yb
            for f in range(F):
                a -= alpha[j, f, levels[i, p, f]]
            for c in range(P):
                a -= X[i, p, c] * gamma[j, c]
            total[j] += om * a
            size[j] += om
            for f in range(F):
                level_count[j, f, levels[i, p, f]] += om
            for c in range(P):
                treated_x[j, c] += om * X[i, p, c]
            for r in range(K):
                pr = om * psi[i, r]
                S[j, r] += pr * a
                slope[j, r] += om * dpsi[i, r]
                for f in range(F):
                    level_basis[j, f, levels[i, p, f], r] += pr
                for c in range(K):
                    A[j, r, c] += pr * psi[i, c]
                for c in range(P):
                    basis_x[j, r, c] += pr * X[i, p, c]


@njit(cache=True, parallel=True)
def general_scores(
    gidx,
    kept,
    y,
    X,
    levels,
    w,
    psi,
    dpsi,
    gi,
    ti,
    bases,
    period,
    controls,
    gamma,
    alpha,
    beta,
    att,
    acrt,
    sizes,
    ainv,
    dbar,
    level_weight,
    slope_weight,
    att_level_weight,
    att_slope_weight,
    local,
    width,
):
    """Each unit's scores for the general (weighted, adjusted) dose fit, in sum scale.

    The layout is that of ``dose_scores`` (binarized ATT, ACRT, the cells' dose
    coefficients, and a block for the control mean change, here always zero). Every
    quantity is per cell: ``sizes[j]`` is the cell's treated weight, ``ainv[j]`` its
    inverse dose moments, ``dbar[j]`` its treated units' average basis slope. A kept
    treated unit observed in the cell scores on its adjusted change ``a`` with weight
    ``w[i, ti[j]]``; a comparison unit with residual ``a`` moves the cell's
    coefficients by ``-w a (sum_f level_weight[f, level_f] + slope_weight X)``
    (``att_*`` the same for the binarized ATT). Units not observed in both of a cell's periods have
    no scores in it, and treated units that are not ``kept`` none in their cohort's.
    """
    B = y.shape[0]
    M = len(gi)
    K = psi.shape[1]
    P = X.shape[2]
    F = levels.shape[2]
    out = np.zeros((B, width))
    first_beta = 2 * M
    for i in prange(B):
        k = gidx[i]
        for j in range(M):
            own = k == gi[j]
            if own and not kept[i]:
                continue
            if not own and not controls[j, k]:
                continue
            t, b, p = ti[j], bases[j], period[j]
            yt, yb = y[i, t], y[i, b]
            if not (np.isfinite(yt) and np.isfinite(yb)):
                continue
            om = w[i, t]
            a = yt - yb
            for f in range(F):
                a -= alpha[j, f, levels[i, p, f]]
            for c in range(P):
                a -= X[i, p, c] * gamma[j, c]
            column = first_beta + j * K
            if own:
                out[i, local[k, j]] = om * (a - att[j]) / sizes[j]
                fit = 0.0
                slope = 0.0
                for c in range(K):
                    fit += psi[i, c] * beta[j, c]
                    slope += dpsi[i, c] * beta[j, c]
                e = om * (a - fit)
                projected = 0.0
                for c in range(K):
                    u = 0.0
                    for d in range(K):
                        u += ainv[j, c, d] * psi[i, d]
                    u *= e
                    out[i, local[k, column + c]] = u
                    projected += dbar[j, c] * u
                out[i, local[k, M + j]] = om * (slope - acrt[j]) / sizes[j] + projected
            else:
                r = om * a
                v0 = 0.0
                for f in range(F):
                    v0 += att_level_weight[j, f, levels[i, p, f]]
                for c in range(P):
                    v0 += att_slope_weight[j, c] * X[i, p, c]
                out[i, local[k, j]] = -r * v0
                projected = 0.0
                for c in range(K):
                    v = 0.0
                    for f in range(F):
                        v += level_weight[j, f, levels[i, p, f], c]
                    for d in range(P):
                        v += slope_weight[j, c, d] * X[i, p, d]
                    out[i, local[k, column + c]] = -r * v
                    projected += dbar[j, c] * v
                out[i, local[k, M + j]] = -r * projected
    return out

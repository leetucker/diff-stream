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


@njit(cache=True)
def comparison_sums(gidx, y, X, levels, periods, n, sx, sy, xx, xy):
    """Sums of the comparison regressions, by cohort, covariate period and level.

    For each unit (cohort ``gidx``) and each covariate period ``periods[q]``, with
    covariates ``X[:, p]`` and fixed-effect level ``levels[:, p]`` at that period:
    ``n[k, q, l]`` counts the units, ``sx[k, q, l]`` sums their covariates and
    ``sy[k, q, l, s]`` their outcome in each period s; ``xx[k, q]`` sums the
    covariates' outer products and ``xy[k, q, :, s]`` their products with the outcome.
    """
    B, T = y.shape
    P = X.shape[2]
    for i in range(B):
        k = gidx[i]
        for q in range(len(periods)):
            p = periods[q]
            lv = levels[i, p]
            n[k, q, lv] += 1.0
            for s in range(T):
                sy[k, q, lv, s] += y[i, s]
            for a in range(P):
                xa = X[i, p, a]
                sx[k, q, lv, a] += xa
                for b in range(P):
                    xx[k, q, a, b] += xa * X[i, p, b]
                for s in range(T):
                    xy[k, q, a, s] += xa * y[i, s]


@njit(cache=True, parallel=True)
def adjusted_scores(
    gidx,
    kept,
    y,
    X,
    levels,
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
    """Each unit's scores for the covariate-adjusted dose fit, in sum scale.

    The layout is that of ``dose_scores`` (binarized ATT, ACRT, the cells' dose
    coefficients, and a block for the control mean change, here always zero). A
    treated unit's change is adjusted by its cell's comparison regression,
    ``a = dY - X'gamma - alpha[level]``; a kept treated unit scores as in the
    unadjusted fit, on ``a``. A comparison unit with residual ``e`` moves the cell's
    dose coefficients by ``-e (level_weight[level] + slope_weight X)``, which carries
    the estimation of the regression into them (``att_*`` the same for the binarized
    ATT). Treated units that are not ``kept`` have no scores in their cohort's cells.
    """
    B = y.shape[0]
    M = len(gi)
    K = psi.shape[1]
    P = X.shape[2]
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
            p = period[j]
            lv = levels[i, p]
            a = y[i, ti[j]] - y[i, bases[j]] - alpha[j, lv]
            for c in range(P):
                a -= X[i, p, c] * gamma[j, c]
            column = first_beta + j * K
            if own:
                out[i, local[k, j]] = (a - att[j]) / sizes[k]
                fit = 0.0
                slope = 0.0
                for c in range(K):
                    fit += psi[i, c] * beta[j, c]
                    slope += dpsi[i, c] * beta[j, c]
                e = a - fit
                projected = 0.0
                for c in range(K):
                    u = 0.0
                    for d in range(K):
                        u += ainv[k, c, d] * psi[i, d]
                    u *= e
                    out[i, local[k, column + c]] = u
                    projected += dbar[k, c] * u
                out[i, local[k, M + j]] = (slope - acrt[j]) / sizes[k] + projected
            else:
                g = gi[j]
                w = att_level_weight[j, lv]
                for c in range(P):
                    w += att_slope_weight[j, c] * X[i, p, c]
                out[i, local[k, j]] = -a * w
                projected = 0.0
                for c in range(K):
                    v = level_weight[j, lv, c]
                    for d in range(P):
                        v += slope_weight[j, c, d] * X[i, p, d]
                    out[i, local[k, column + c]] = -a * v
                    projected += dbar[g, c] * v
                out[i, local[k, M + j]] = -a * projected
    return out

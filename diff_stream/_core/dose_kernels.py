"""Unit influence scores of the continuous-treatment (dose) DiD, one batch at a time."""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def dose_scores(
    gidx,
    y,
    psi,
    dpsi,
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
    ``dpsi`` are the units' basis values and derivatives (zero if never treated).
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
                slope = 0.0
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
    return out

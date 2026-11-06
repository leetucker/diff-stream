"""Balanced-panel algebra and bounded, parallel cluster-score kernels.

In a saturated SA regression with a never-treated reference, the within-unit
normal equations reduce exactly to cohort-versus-never changes from g-1.
This avoids a dense cohort-interaction regression system altogether.
"""

import numpy as np
from numba import njit, prange


def setup(method, times, gc, control_group, base_period, anticipation=0):
    groups = sorted(gc)
    cells, bases, controls = [], [], []
    for g in groups[1:]:
        onset = g - anticipation
        for t in times:
            if (method == "bjs" and t < onset) or (
                t == onset - 1 and (method == "sunab" or base_period == "universal")
            ):
                continue
            b = (
                int(t - 1)
                if method == "cs" and base_period == "varying" and t < onset
                else onset - 1
            )
            if b < times[0]:
                continue
            c = [
                h == 0
                or (control_group == "notyettreated" and h - anticipation > max(t, b) and h != g)
                for h in groups
            ]
            cells.append((g, int(t)))
            bases.append(b - times[0])
            controls.append(c)
    return (
        groups,
        cells,
        np.array(bases, dtype=np.int64),
        np.array(controls, dtype=np.bool_),
        np.array([gc[g] for g in groups], dtype=float),
    )


def fit_means(method, times, groups, cells, bases, controls, sizes, means, anticipation=0):
    T = len(times)
    gi = np.array([groups.index(g) for g, t in cells], dtype=np.int64)
    ti = np.array([t - times[0] for g, t in cells], dtype=np.int64)
    counts = sizes[gi]
    if method != "bjs":
        nc = controls @ sizes
        control_mean = np.sum(controls * sizes * (means[:, ti].T - means[:, bases].T), axis=1) / nc
        att = means[gi, ti] - means[gi, bases] - control_mean
        return att, counts, (gi, ti, bases, controls, nc, means, control_mean)
    # Unit effects are eliminated exactly on untreated rows. Only T-1 time
    # coefficients enter the solve. Normalize the first time effect to zero.
    Z = np.ascontiguousarray(np.eye(T)[:, 1:])
    Q, q = np.zeros((T - 1, T - 1)), np.zeros(T - 1)
    zbar, alpha = [], []
    for k, g in enumerate(groups):
        use = (times < g - anticipation) if g else np.ones(T, dtype=bool)
        z = Z[use]
        zm = z.mean(axis=0)
        zbar.append(zm)
        zd = z - zm
        Q += sizes[k] * zd.T @ zd
        q += sizes[k] * zd.T @ means[k, use]
    theta = np.linalg.solve(Q, q)
    for k, g in enumerate(groups):
        use = (times < g - anticipation) if g else np.ones(T, dtype=bool)
        alpha.append(np.mean(means[k, use] - Z[use] @ theta))
    att = means[gi, ti] - np.array(alpha)[gi] - Z[ti] @ theta
    H = np.linalg.solve(Q, (Z[ti] - np.array(zbar)[gi]).T).T
    return att, counts, (gi, ti, Z, theta, H)


@njit(cache=True, parallel=True)
def change_scores(gidx, y, gi, ti, bases, controls, nc, means, control_mean, sizes):
    B, T = y.shape
    M = len(gi)
    scores = np.zeros((B, M))
    shares = np.zeros((B, M))
    for i in prange(B):
        k = gidx[i]
        for j in range(M):
            change = y[i, ti[j]] - y[i, bases[j]]
            if k == gi[j]:
                scores[i, j] = (change - means[k, ti[j]] + means[k, bases[j]]) / sizes[k]
                shares[i, j] = 1.0
            elif controls[j, k]:
                # With changing controls, center over the entire comparison
                # population, not separately within its constituent cohorts.
                scores[i, j] = -(change - control_mean[j]) / nc[j]
    return scores, shares

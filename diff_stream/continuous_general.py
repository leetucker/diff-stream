"""The general fit of did_continuous: one pass structure for every option beyond contdid's.

It is used with covariates or a fixed effect, the lowest-dose comparison, weights,
an unbalanced panel or clustering above the unit; without any of them the fit in
``continuous`` is used, and this one gives the same results.

In each cohort-period cell j (base period b, covariate period p = min(t, b)), the
comparison units observed in both periods have their outcome change regressed, with
their weights in period t, on the covariates at p and their fixed-effect level at p.
The regression is solved exactly from weighted sums by level (Frisch-Waugh-Lovell:
the covariates are demeaned within levels), so a fixed effect with many levels needs
no iterative solver. Without covariates or a fixed effect it is the comparison units'
weighted mean change. Each treated unit observed in the cell has its change less the
prediction fitted on the dose basis: beta_j = A_j^-1 sum w psi a, with
A_j = sum w psi psi'. ATT(d) = psi(d)'beta_j, ACRT(d) = psi'(d)'beta_j, and the
binarized ATT is the weighted mean of a. Everything is per cell, so the units and
weights may differ between cells.

With the lowest-dose comparison, the units at the reference doses are relabeled as
the comparison group (group 0) and the others form the cohort; nothing else changes.

Treated units whose level has no comparison units in a cell of their cohort they are
observed in cannot be adjusted: they are left out of their cohort's estimates, and
remain comparison units for the others. Scores are summed by unit, or by cluster
when clustering above the unit.
"""

import numpy as np

from ._core.dose_kernels import (
    cell_comparison_sums,
    cell_treated_sums,
    general_scores,
    kept_units,
)
from ._core.multiplier import StreamingMultiplier
from ._core.nuisance import solve_checked
from .cohort_scores import CohortScoreCovariance


def table_bytes(M, L, P, K, C, W):
    """Memory of the per-cell tables and the cluster sums of the general fit."""
    cells = M * L * (P + 3 + 2 * K) + M * (K * (P + 2 * K + 4) + P * (P + 3))
    return 8 * (cells + C * (W + 1))


class ClusterScores:
    """Sums of the units' scores and share anchors by cluster.

    ``blocks`` returns (V, cross, shares) as ``CohortScoreCovariance`` does, over
    clusters instead of units: V = sum_c S_c S_c', with S_c the cluster's score sum,
    and the share terms from each cluster's anchor sums by cohort.
    """

    def __init__(self, n_clusters, width, n_groups):
        self.S = np.zeros((n_clusters, width))
        self.A = np.zeros((n_clusters, n_groups))

    def add(self, cluster, scores, group, anchor):
        order = np.argsort(cluster, kind="stable")
        c = cluster[order]
        starts = np.r_[0, np.flatnonzero(c[1:] != c[:-1]) + 1]
        self.S[c[starts]] += np.add.reduceat(scores[order], starts, axis=0)
        np.add.at(self.A, (cluster, group), anchor)

    def blocks(self, labels, center):
        H = self.A[:, labels] - center[None, :] * self.A.sum(axis=1)[:, None]
        return self.S.T @ self.S, self.S.T @ H, H.T @ H


def fit_general(
    store,
    basis,
    groups,
    cells,
    bases,
    controls,
    *,
    dose_column,
    fe_column,
    n_levels,
    covariate_columns,
    relabel=None,
    cluster_column=None,
    n_clusters=0,
    bootstrap_reps=0,
    bootstrap_seed=None,
    bootstrap_weights="rademacher",
):
    """The cells, their covariance and the units left out, in the unadjusted fit's layout.

    ``relabel(g, d)`` gives each unit's group from its cohort and dose (the
    lowest-dose comparison); ``cluster_column`` the design column of a cluster index
    above the unit, with ``n_clusters`` levels. Returns a dict with the binarized ATT
    and the ACRT per cell, the cells' dose coefficients, the covariance blocks (V,
    cross and the centered share covariance), the cohorts' weighted sizes and units
    left out, the total anchor weight and the multiplier draws.
    """
    times = store.times
    M, G, T = len(cells), len(groups), len(times)
    K, P, L = basis.width, len(covariate_columns), max(n_levels, 1)
    groups_array = np.asarray(groups)
    gi = np.array([groups.index(g) for g, t in cells], dtype=np.int64)
    ti = np.array([t - times[0] for g, t in cells], dtype=np.int64)
    bases = np.asarray(bases, dtype=np.int64)
    period = np.minimum(ti, bases)
    controls = np.ascontiguousarray(np.asarray(controls, dtype=bool))
    cov = np.asarray(covariate_columns, dtype=np.int64)

    def integer_column(x, column):
        v = x[:, :, column]
        return np.where(np.isfinite(v), v, 0.0).astype(np.int64)

    def unit_value(x, column, what):
        v = x[:, :, column]
        low, high = np.nanmin(v, axis=1), np.nanmax(v, axis=1)
        if np.any(low != high):
            raise ValueError(f"the {what} must be constant within each unit")
        return high

    def arrays(batch):
        g, y, x, w = batch
        d = unit_value(x, dose_column, "dose")
        if relabel is not None:
            g = relabel(g, d)
        k = np.searchsorted(groups_array, g)
        X = np.ascontiguousarray(x[:, :, cov]) if P else np.zeros((len(g), T, 0))
        lv = (
            integer_column(x, fe_column)
            if fe_column is not None
            else np.zeros((len(g), T), dtype=np.int64)
        )
        cl = (
            unit_value(x, cluster_column, "cluster").astype(np.int64)
            if cluster_column is not None
            else None
        )
        observed = np.isfinite(y)
        anchor = w.sum(axis=1) / np.maximum(observed.sum(axis=1), 1)
        return k, np.ascontiguousarray(y), X, lv, d, np.ascontiguousarray(w), cl, anchor

    def batches():
        for batch in store.batches(include_x=True, include_weights=True, row_weights=True):
            yield arrays(batch)

    # Pass A: the comparison regressions' sums, cell by cell.
    n, sy = np.zeros((M, L)), np.zeros((M, L))
    sx, xx, xy = np.zeros((M, L, P)), np.zeros((M, P, P)), np.zeros((M, P))
    for k, y, X, lv, _, w, _, _ in batches():
        cell_comparison_sums(k, y, X, lv, w, gi, ti, bases, period, controls, n, sx, sy, xx, xy)

    gamma, alpha = np.zeros((M, P)), np.zeros((M, L))
    xbar, inverse = np.zeros((M, L, P)), np.zeros((M, P, P))
    for j, (g, t) in enumerate(cells):
        nl = n[j]
        present = nl > 0
        if not present.any():
            raise ValueError(f"cohort {g} has no comparison units observed in period {t}")
        means = np.zeros((L, P))
        means[present] = sx[j][present] / nl[present, None]
        mean_change = np.zeros(L)
        mean_change[present] = sy[j][present] / nl[present]
        if P:
            W = xx[j] - means[present].T @ (nl[present, None] * means[present])
            r = xy[j] - means[present].T @ sy[j][present]
            context = (
                f"the comparison regression of cohort {g}, period {t} (are the covariates "
                "collinear with the fixed effect?)"
            )
            gamma[j] = solve_checked(W, r, context)
            inverse[j] = solve_checked(W, np.eye(P), context)
        alpha[j] = np.where(present, mean_change - means @ gamma[j], 0.0)
        xbar[j] = means

    # Pass B: the treated units' moments on their adjusted changes, cell by cell.
    A, S = np.zeros((M, K, K)), np.zeros((M, K))
    total, size, slope = np.zeros(M), np.zeros(M), np.zeros((M, K))
    level_basis, level_count = np.zeros((M, L, K)), np.zeros((M, L))
    basis_x, treated_x = np.zeros((M, K, P)), np.zeros((M, P))
    left_out, cohort_weight = np.zeros(G, dtype=np.int64), np.zeros(G)
    total_weight = 0.0

    def treated_rows(k, y, lv, d):
        treated = groups_array[k] != 0
        kept = kept_units(k, treated, y, lv, gi, ti, bases, period, n)
        Pm, Dm = np.zeros((len(k), K)), np.zeros((len(k), K))
        use = treated & kept
        if use.any():
            Pm[use], Dm[use] = basis.values(d[use]), basis.slopes(d[use])
        return treated, kept, Pm, Dm

    for k, y, X, lv, d, w, _, anchor in batches():
        treated, kept, Pm, Dm = treated_rows(k, y, lv, d)
        np.add.at(left_out, k[treated & ~kept], 1)
        np.add.at(cohort_weight, k[kept], anchor[kept])
        total_weight += float(anchor[kept].sum())
        cell_treated_sums(
            k,
            kept,
            y,
            X,
            lv,
            w,
            Pm,
            Dm,
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
        )

    for c, g in enumerate(groups):
        if g != 0 and cohort_weight[c] == 0:
            raise ValueError(
                f"no unit of cohort {g} has comparison units in its fixed-effect level"
            )
    ainv = np.zeros((M, K, K))
    for j, (g, t) in enumerate(cells):
        if size[j] == 0:
            raise ValueError(f"cohort {g} has no units observed in period {t} and its base")
        if np.linalg.cond(A[j]) > 1e12:
            raise ValueError(
                f"the dose basis is not identified in cohort {g}, period {t}: too few "
                "distinct doses among its units; reduce degree or interior_knots"
            )
        ainv[j] = np.linalg.inv(A[j])
    dbar = slope / size[:, None]
    beta = np.einsum("jab,jb->ja", ainv, S)
    att = total / size
    acrt = np.einsum("jk,jk->j", dbar, beta)

    # How a comparison unit's residual moves each cell's coefficients.
    level_weight, slope_weight = np.zeros((M, L, K)), np.zeros((M, K, P))
    att_level_weight, att_slope_weight = np.zeros((M, L)), np.zeros((M, P))
    for j in range(M):
        nl = n[j]
        present = nl > 0
        if P:
            H = ainv[j] @ (basis_x[j] - level_basis[j].T @ xbar[j]) @ inverse[j]
            h = (treated_x[j] - level_count[j] @ xbar[j]) @ inverse[j] / size[j]
            slope_weight[j], att_slope_weight[j] = H, h
        else:
            H, h = np.zeros((K, 0)), np.zeros(0)
        level_weight[j][present] = (level_basis[j][present] @ ainv[j].T) / nl[present, None]
        level_weight[j] -= xbar[j] @ H.T
        att_level_weight[j][present] = level_count[j][present] / (size[j] * nl[present])
        att_level_weight[j] -= xbar[j] @ h

    # Pass C: the units' scores, in the layout of the unadjusted fit.
    W = M * (K + 3)
    labels = np.r_[gi, gi, np.repeat(gi, K), gi]
    share = cohort_weight[gi] / total_weight
    center = np.r_[share, share, np.repeat(share, K), share]
    identity = np.broadcast_to(np.arange(W), (G, W)).copy()
    if cluster_column is not None:
        sums = ClusterScores(n_clusters, W, G)
    else:
        covariance = CohortScoreCovariance(labels, center)
        columns = {}
        for c in range(G):
            own, control = np.flatnonzero(gi == c), np.flatnonzero(controls[:, c])
            both = np.union1d(own, control)
            columns[c] = np.sort(
                np.r_[both, M + both, (2 * M + K * both[:, None] + np.arange(K)).ravel()]
            )
        local = np.zeros((G, W), dtype=np.int64)
        for c, used in columns.items():
            local[c, used] = np.arange(len(used))
        compact = max(len(used) for used in columns.values())
    multiplier = (
        StreamingMultiplier(M * K + M, bootstrap_reps, bootstrap_seed, bootstrap_weights)
        if bootstrap_reps
        else None
    )
    common = (
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
        size,
        ainv,
        dbar,
        level_weight,
        slope_weight,
        att_level_weight,
        att_slope_weight,
    )
    for k, y, X, lv, d, w, cl, anchor in batches():
        _, kept, Pm, Dm = treated_rows(k, y, lv, d)
        unit_anchor = anchor * kept
        if cluster_column is not None:
            full = general_scores(k, kept, y, X, lv, w, Pm, Dm, *common, identity, W)
            sums.add(cl, full, k, unit_anchor)
            continue
        order = np.argsort(k, kind="stable")
        ko = k[order]
        scores = general_scores(
            ko,
            kept[order],
            y[order],
            X[order],
            lv[order],
            w[order],
            Pm[order],
            Dm[order],
            *common,
            local,
            compact,
        )
        starts = np.r_[0, np.flatnonzero(ko[1:] != ko[:-1]) + 1, len(ko)]
        for a, b in zip(starts[:-1], starts[1:], strict=True):
            c = int(ko[a])
            used = columns[c]
            covariance.add_block(
                c, scores[a:b, : len(used)], used, unit_anchor[order][a:b], count=True
            )
        if multiplier is not None:
            full = general_scores(k, kept, y, X, lv, w, Pm, Dm, *common, identity, W)
            multiplier.update(full[:, 2 * M :])
    if cluster_column is not None:
        V, cross, share_cov = sums.blocks(labels, center)
        if multiplier is not None:
            multiplier.update(sums.S[:, 2 * M :])
    else:
        V, cross, share_cov = covariance.blocks()
    return dict(
        att=att,
        acrt=acrt,
        beta=beta,
        V=V,
        cross=cross,
        share_cov=share_cov,
        sizes={g: float(cohort_weight[c]) for c, g in enumerate(groups)},
        left_out={g: int(left_out[c]) for c, g in enumerate(groups) if g != 0},
        total_weight=total_weight,
        draws=multiplier.draws if multiplier is not None else None,
    )

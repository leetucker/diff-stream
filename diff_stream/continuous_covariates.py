"""Covariates and one absorbed fixed effect in did_continuous (experimental).

In each cohort-period cell, the comparison units' outcome change since the base
period is regressed on the covariates and the fixed effect's levels; each treated
unit's change less its prediction is then fitted on the dose basis as without
covariates. Covariates and the level are those of the cell's covariate period (the
base period, or the period itself when it comes first), as in Callaway-Sant'Anna.

The regression of a cell is solved exactly from sums by level (Frisch-Waugh-Lovell:
the covariates are demeaned within levels), so a fixed effect with many levels needs
no iterative solver. The sums are kept by cohort, covariate period and level, and a
cell's comparison sums are those of its comparison cohorts. Treated units in a level
that has no comparison units in one of their cohort's cells cannot be adjusted: they
are left out of their cohort's estimates, and remain comparison units for the others.
"""

import re

import numpy as np
import polars as pl
from hdfe_stream.core.report import ExperimentalWarning as _Experimental

from ._core.dose_kernels import adjusted_scores, comparison_sums
from ._core.multiplier import StreamingMultiplier
from ._core.nuisance import solve_checked
from .cohort_scores import CohortScoreCovariance

FE = "__continuous_fe"


class ExperimentalWarning(_Experimental):
    """A diff-stream fit uses a method its authors have not developed.

    It subclasses ``hdfe_stream.ExperimentalWarning``: filter this class to silence
    diff-stream's experimental warnings, or that one for every package built on
    hdfe-stream.
    """


EXPERIMENTAL = (
    "Covariates in did_continuous are an extension of Callaway, Goodman-Bacon and "
    "Sant'Anna that the authors have not developed. ATT(d) requires parallel trends "
    "conditional on the covariates; ACRT(d) also requires strong parallel trends given "
    "the covariates, and is not an average causal response if the covariates change "
    "with the dose. See "
    "https://leetucker.github.io/diff-stream/estimators/continuous.html#covariates-experimental"
)


def parse(fml, dose, panel):
    """The outcome, the covariates' formula (or None) and the fixed effect's columns (or None)."""
    if not isinstance(fml, str) or fml.count("~") != 1:
        raise ValueError("use one outcome formula: 'y ~ 1', 'y ~ x' or 'y ~ x | region'")
    y, rhs = [s.strip() for s in fml.split("~")]
    if not y.isidentifier():
        raise ValueError("the outcome must be a column name")
    parts = [s.strip() for s in rhs.split("|")]
    if len(parts) > 2:
        raise ValueError("give at most one '|' (the fixed effect): 'y ~ x | region'")
    covariates = parts[0] if parts[0] not in ("1", "") else None
    if covariates is not None and dose in re.findall(r"[A-Za-z_]\w*", covariates):
        raise ValueError("the dose enters through dose=, not as a covariate")
    fe = None
    if len(parts) == 2:
        term = parts[1]
        if "+" in term:
            raise ValueError(
                "did_continuous absorbs one fixed-effect term (which may be an "
                "interaction, region^industry); give the others as covariates"
            )
        fe = [c.strip() for c in re.split(r"[\^:]", term)]
        if not all(c.isidentifier() for c in fe) or len(set(fe)) != len(fe):
            raise ValueError(f"cannot read the fixed effect {term!r}")
        unit, time = panel
        if unit in fe:
            raise ValueError(
                f"{unit} (the unit) cannot be a fixed effect: unit effects are already "
                "differenced out"
            )
        if time in fe:
            raise ValueError(
                f"{time} (the period) cannot be a fixed effect: each cohort-period has "
                "its own regression, so a fixed effect already differs by period "
                "(| region is region-by-period)"
            )
        if dose in fe or y in fe:
            raise ValueError("the fixed effect cannot be the dose or the outcome")
    return y, covariates, fe


def join_levels(src, columns):
    """The source with each distinct value of ``columns`` numbered as ``FE``, and how many."""
    missing = set(columns) - set(src.collect_schema())
    if missing:
        raise ValueError(f"unknown fixed-effect columns: {sorted(missing)}")
    levels = src.select(columns).unique().collect(engine="streaming")
    if levels.null_count().sum_horizontal().item():
        raise ValueError("the fixed effect has missing values; fill or drop them first")
    levels = levels.sort(columns).with_row_index(FE).with_columns(pl.col(FE).cast(pl.Float64))
    return src.join(levels.lazy(), on=columns, how="left"), len(levels)


def table_bytes(G, Q, L, P, T, M, K):
    """Memory of the sums and per-cell tables of the adjusted fit."""
    sums = G * Q * (L * (1 + P + T) + P * (P + T))
    cells = M * L * (P + 3 + 2 * K) + M * K * (P + K + 2)
    return 8 * (sums + cells)


def fit_adjusted(
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
    bootstrap_reps,
    bootstrap_seed,
    bootstrap_weights,
):
    """The covariate-adjusted cells, their covariance and the units left out.

    Returns a dict with the binarized ATT and the ACRT per cell, the cells' dose
    coefficients, the joint covariance blocks (V, cross, shares) in the layout of the
    unadjusted fit, the cohorts' sizes and units left out, and the multiplier draws.
    """
    times = store.times
    M, G, T = len(cells), len(groups), len(times)
    K, P, L = basis.width, len(covariate_columns), max(n_levels, 1)
    gi = np.array([groups.index(g) for g, t in cells], dtype=np.int64)
    ti = np.array([t - times[0] for g, t in cells], dtype=np.int64)
    bases = np.asarray(bases, dtype=np.int64)
    # the covariate period of each cell, as Callaway-Sant'Anna's: the earlier of the
    # period and its base; one period when nothing in the design changes over time
    period = np.zeros(M, dtype=np.int64) if store.x_time_invariant else np.minimum(ti, bases)
    distinct = np.unique(period)
    where = np.searchsorted(distinct, period)
    controls = np.asarray(controls, dtype=bool)
    cov = np.asarray(covariate_columns, dtype=np.int64)

    def arrays(g, y, x):
        k = np.searchsorted(groups, g)
        X = np.ascontiguousarray(x[:, :, cov]) if P else np.zeros((len(g), T, 0))
        lv = (
            x[:, :, fe_column].astype(np.int64)
            if fe_column is not None
            else np.zeros((len(g), T), dtype=np.int64)
        )
        return k, np.ascontiguousarray(y), X, lv, x[:, 0, dose_column]

    # Pass A: the comparison regressions' sums.
    Q = len(distinct)
    n = np.zeros((G, Q, L))
    sx, sy = np.zeros((G, Q, L, P)), np.zeros((G, Q, L, T))
    xx, xy = np.zeros((G, Q, P, P)), np.zeros((G, Q, P, T))
    for g, y, x in store.batches(include_x=True):
        if np.any(x[:, :, dose_column] != x[:, :1, dose_column]):
            raise ValueError("the dose must be constant within each unit")
        k, y, X, lv, _ = arrays(g, y, x)
        comparison_sums(k, y, X, lv, distinct, n, sx, sy, xx, xy)

    gamma = np.zeros((M, P))
    alpha, count, xbar = np.zeros((M, L)), np.zeros((M, L)), np.zeros((M, L, P))
    inverse = np.zeros((M, P, P))
    for j, (g, t) in enumerate(cells):
        q, use = where[j], controls[j]
        nl = n[use, q].sum(axis=0)
        sxl = sx[use, q].sum(axis=0)
        change = sy[use, q, :, ti[j]].sum(axis=0) - sy[use, q, :, bases[j]].sum(axis=0)
        present = nl > 0
        means = np.zeros((L, P))
        means[present] = sxl[present] / nl[present, None]
        mean_change = np.zeros(L)
        mean_change[present] = change[present] / nl[present]
        if P:
            W = xx[use, q].sum(axis=0) - means[present].T @ (nl[present, None] * means[present])
            r = xy[use, q][:, :, ti[j]].sum(axis=0) - xy[use, q][:, :, bases[j]].sum(axis=0)
            r = r - means[present].T @ change[present]
            context = (
                f"the comparison regression of cohort {g}, period {t} (are the covariates "
                "collinear with the fixed effect?)"
            )
            gamma[j] = solve_checked(W, r, context)
            inverse[j] = solve_checked(W, np.eye(P), context)
        alpha[j] = np.where(present, mean_change - means @ gamma[j], 0.0)
        count[j], xbar[j] = nl, means

    # Pass B: the treated units' adjusted moments. A treated unit is left out of its
    # cohort's cells if its level has no comparison units in one of them.
    mine = [np.flatnonzero(gi == k) for k in range(G)]
    A, S = np.zeros((G, K, K)), np.zeros((M, K))
    slope_sums, sizes = np.zeros((G, K)), np.zeros(G)
    adjusted_sums = np.zeros(M)
    level_basis, level_count = np.zeros((M, L, K)), np.zeros((M, L))
    basis_x, treated_x = np.zeros((M, K, P)), np.zeros((M, P))
    left_out = np.zeros(G, dtype=np.int64)

    def adjusted(j, y, X, lv):
        p = period[j]
        change = y[:, ti[j]] - y[:, bases[j]]
        return change - X[:, p] @ gamma[j] - alpha[j, lv[:, p]]

    def kept_rows(k, lv):
        ok = np.ones(len(lv), dtype=bool)
        for j in mine[k]:
            ok &= count[j, lv[:, period[j]]] > 0
        return ok

    for g, y, x in store.batches(include_x=True):
        k, y, X, lv, d = arrays(g, y, x)
        for c in np.unique(k):
            if groups[c] == 0:
                continue
            rows = np.flatnonzero(k == c)
            ok = kept_rows(c, lv[rows])
            left_out[c] += int((~ok).sum())
            rows = rows[ok]
            if not len(rows):
                continue
            Pm, Dm = basis.values(d[rows]), basis.slopes(d[rows])
            A[c] += Pm.T @ Pm
            slope_sums[c] += Dm.sum(axis=0)
            sizes[c] += len(rows)
            for j in mine[c]:
                a = adjusted(j, y[rows], X[rows], lv[rows])
                S[j] += Pm.T @ a
                adjusted_sums[j] += a.sum()
                levels_j = lv[rows, period[j]]
                np.add.at(level_basis[j], levels_j, Pm)
                level_count[j] += np.bincount(levels_j, minlength=L)
                if P:
                    Xj = X[rows, period[j]]
                    basis_x[j] += Pm.T @ Xj
                    treated_x[j] += Xj.sum(axis=0)

    ainv = np.zeros((G, K, K))
    for c, g in enumerate(groups):
        if g == 0:
            continue
        if sizes[c] == 0:
            raise ValueError(
                f"no unit of cohort {g} has comparison units in its fixed-effect level"
            )
        if np.linalg.cond(A[c]) > 1e12:
            raise ValueError(
                f"the dose basis is not identified in cohort {g}: too few distinct doses "
                "among its units"
            )
        ainv[c] = np.linalg.inv(A[c])
    dbar = slope_sums / np.maximum(sizes, 1)[:, None]
    beta = np.einsum("jab,jb->ja", ainv[gi], S)
    att = adjusted_sums / sizes[gi]
    acrt = np.einsum("jk,jk->j", dbar[gi], beta)

    # How a comparison unit's residual moves each cell's coefficients.
    level_weight, slope_weight = np.zeros((M, L, K)), np.zeros((M, K, P))
    att_level_weight, att_slope_weight = np.zeros((M, L)), np.zeros((M, P))
    for j in range(M):
        c, nl = gi[j], count[j]
        present = nl > 0
        if P:
            H = ainv[c] @ (basis_x[j] - level_basis[j].T @ xbar[j]) @ inverse[j]
            h = (treated_x[j] - level_count[j] @ xbar[j]) @ inverse[j] / sizes[c]
            slope_weight[j], att_slope_weight[j] = H, h
        else:
            H, h = np.zeros((K, 0)), np.zeros(0)
        level_weight[j][present] = (level_basis[j][present] @ ainv[c].T) / nl[present, None]
        level_weight[j] -= xbar[j] @ H.T
        att_level_weight[j][present] = level_count[j][present] / (sizes[c] * nl[present])
        att_level_weight[j] -= xbar[j] @ h

    # Pass C: the units' scores, in the layout of the unadjusted fit.
    W = M * (K + 3)
    labels = np.r_[gi, gi, np.repeat(gi, K), gi]
    covariance = CohortScoreCovariance(labels)
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
    identity = np.broadcast_to(np.arange(W), (G, W)).copy()
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
        np.maximum(sizes, 1),
        ainv,
        dbar,
        level_weight,
        slope_weight,
        att_level_weight,
        att_slope_weight,
    )
    for g, y, x in store.batches(include_x=True):
        k, y, X, lv, d = arrays(g, y, x)
        kept = np.ones(len(g), dtype=bool)
        for c in np.unique(k):
            if groups[c] != 0:
                rows = np.flatnonzero(k == c)
                kept[rows] = kept_rows(c, lv[rows])
        order = np.argsort(k, kind="stable")
        k, y, X, lv, d, kept = k[order], y[order], X[order], lv[order], d[order], kept[order]
        Pm, Dm = np.zeros((len(k), K)), np.zeros((len(k), K))
        treated = (np.asarray(groups)[k] != 0) & kept
        if treated.any():
            Pm[treated], Dm[treated] = basis.values(d[treated]), basis.slopes(d[treated])
        scores = adjusted_scores(k, kept, y, X, lv, Pm, Dm, *common, local, compact)
        starts = np.r_[0, np.flatnonzero(k[1:] != k[:-1]) + 1, len(k)]
        for a, b in zip(starts[:-1], starts[1:], strict=True):
            c = int(k[a])
            used = columns[c]
            # units left out of their cohort do not count in its share
            anchor = kept[a:b].astype(float) if groups[c] != 0 else None
            covariance.add_block(c, scores[a:b, : len(used)], used, anchor, count=True)
        if multiplier is not None:
            full = adjusted_scores(k, kept, y, X, lv, Pm, Dm, *common, identity, W)
            multiplier.update(full[np.argsort(order), 2 * M :])
    V, cross, shares = covariance.blocks()
    return dict(
        att=att,
        acrt=acrt,
        beta=beta,
        V=V,
        cross=cross,
        shares=shares,
        sizes={g: int(sizes[c]) for c, g in enumerate(groups)},
        left_out={g: int(left_out[c]) for c, g in enumerate(groups) if g != 0},
        draws=multiplier.draws if multiplier is not None else None,
    )

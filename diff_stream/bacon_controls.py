"""Covariate-adjusted Bacon identity via weighted orthogonal projections.

Unit/time/covariate residuals split into within-timing-group variation and
cohort-time means. Pairwise mean contrasts recover the between component;
no unit-by-pair table or pair-specific source regression is required.
"""

from itertools import combinations

import numpy as np
import polars as pl

from ._core.nuisance import solve_checked
from ._core.within import center_groups, independent_columns


def decompose_controls(store, columns):
    T = store.periods
    P = T - 1 + len(columns)
    groups = sorted(store.gc)
    index = {g: j for j, g in enumerate(groups)}
    masses = np.array([store.gw[g] for g in groups])
    total = masses.sum()
    physical = np.array([store.gc[g] for g in groups])
    rawmass = masses * store.weight_scale / store.weight_normalizer
    time = np.eye(T)[:, 1:]

    def batches():
        for g, y, x, w in store.batches(include_x=True, include_weights=True, row_weights=True):
            X = np.concatenate([np.broadcast_to(time, (*y.shape, T - 1)), x[:, :, columns]], axis=2)
            D = ((g[:, None] != 0) & (store.times >= g[:, None])).astype(float)
            xc, yc, _, _ = center_groups(X, y, w)
            dc = D - D.mean(axis=1, keepdims=True)
            yield g, xc, yc, dc, w, X

    Q, q, norm = np.zeros((P, P)), np.zeros(P), np.zeros(P)
    for _, X, _, D, w, raw in batches():
        z = X.reshape(-1, P)
        sw = w.ravel()
        Q += z.T @ (sw[:, None] * z)
        q += z.T @ (sw * D.ravel())
        norm += np.einsum("btp,bt,btp->p", raw, w, raw, optimize=True)
    keep = independent_columns(Q, original_norm=norm)
    coef = np.zeros(P)
    if len(keep):
        coef[keep] = solve_checked(Q[np.ix_(keep, keep)], q[keep], "Bacon treatment projection")
    rmean = np.zeros((len(groups), T))
    ymean = np.zeros_like(rmean)
    rr, ry = 0.0, 0.0
    for g, X, y, D, w, _ in batches():
        r = D - X @ coef
        rr += float(np.sum(w * r * r))
        ry += float(np.sum(w * r * y))
        for cg in np.unique(g):
            mask = g == cg
            rmean[index[cg]] += np.sum(w[mask] * r[mask], axis=0) / masses[index[cg]]
            ymean[index[cg]] += np.sum(w[mask] * y[mask], axis=0) / masses[index[cg]]
    if not np.isfinite(rr) or rr <= 1e-20 * max(total * T, 1):
        raise ValueError(
            "treatment is collinear with unit/time effects and covariates; no identified TWFE decomposition"
        )
    within_rr, within_ry = 0.0, 0.0
    for g, X, y, D, w, _ in batches():
        r = D - X @ coef
        rw = r - np.stack([rmean[index[cg]] for cg in g])
        within_rr += float(np.sum(w * rw * rw))
        within_ry += float(np.sum(w * rw * y))
    # Exact zero components can pick up rounding noise after group reduction.
    if within_rr <= rr * 1e-24 and abs(within_ry) <= 1e-12 * max(abs(ry), 1):
        within_rr, within_ry = 0.0, 0.0
    omega = within_rr / rr
    beta_within = within_ry / within_rr if within_rr > 0 else None
    within_contribution = within_ry / rr
    if omega > 1 + 1e-10:
        raise FloatingPointError(
            "Bacon within/between treatment projection lost numerical precision"
        )
    rows = []
    first = int(store.times[0])
    for a, b in combinations(groups, 2):
        i, j = index[a], index[b]
        dr = rmean[i] - rmean[j]
        dy = ymean[i] - ymean[j]
        pair_rr = float(dr @ dr)
        pair_ry = float(dr @ dy)
        raw = masses[i] * masses[j] / total * pair_rr
        weight = raw / rr
        if pair_rr <= 1e-24 * rr / total:
            estimate = None
            weight = 0.0
            contribution = 0.0
        else:
            estimate = pair_ry / pair_rr
            contribution = weight * estimate
        if a == 0 or b == 0:
            treated = b if a == 0 else a
            control = 0
            kind = "always_vs_never" if treated == first else "treated_vs_never"
        elif a == first or b == first:
            treated = b if a == first else a
            control = first
            kind = "later_vs_always"
        else:
            treated, control, kind = a, b, "both_treated"
        ti, ci = index[treated], index[control]
        rows.append(
            dict(
                treated_cohort=int(treated),
                control_cohort=int(control),
                comparison_type=kind,
                control_already_treated=True
                if kind == "later_vs_always"
                else None
                if kind == "both_treated"
                else False,
                pre_start=None,
                pre_end=None,
                post_start=None,
                post_end=None,
                pre_periods=None,
                post_periods=None,
                n_treated_units=int(physical[ti]),
                n_control_units=int(physical[ci]),
                treated_weight_total=float(rawmass[ti]),
                control_weight_total=float(rawmass[ci]),
                estimate=float(estimate) if estimate is not None else None,
                weight=float(weight),
                contribution=float(contribution),
            )
        )
    between = sum(row["weight"] for row in rows)
    for row in rows:
        row["between_weight"] = row["weight"] / between if between > 0 else None
    twfe = ry / rr
    reconstructed = within_contribution + sum(row["contribution"] for row in rows)
    scale = max(
        1.0,
        abs(twfe),
        max((abs(row["estimate"]) for row in rows if row["estimate"] is not None), default=0.0),
    )
    if not np.isfinite(twfe) or not np.isfinite(reconstructed):
        raise FloatingPointError(
            "covariate-adjusted Bacon calculation overflowed; rescale the outcome"
        )
    if not np.isclose(between + omega, 1, atol=1e-10, rtol=0) or not np.isclose(
        reconstructed, twfe, atol=1e-10 * scale, rtol=0
    ):
        raise FloatingPointError(
            "covariate-adjusted Bacon reconstruction lost numerical precision; rescale covariates/outcome"
        )
    names = [f"time::{t}" for t in store.times[1:]] + [store.design.names[j] for j in columns]
    metadata = dict(
        scope="balanced covariate-adjusted unit/time fixed effects; within-timing plus full-period timing-group dyads",
        within_timing_weight=float(omega),
        between_timing_weight=float(between),
        comparison_weights="total contribution weights; between_weight additionally normalizes dyads alone",
        residualized_treatment_variance=rr / (total * T),
        nuisance_columns=[names[j] for j in keep],
        absorbed_or_redundant_columns=[name for j, name in enumerate(names) if j not in keep],
        comparison_interpretation="full-period residualized timing-group contrasts; both_treated combines clean/forbidden comparisons",
    )
    return (
        pl.DataFrame(
            rows,
            schema_overrides={
                "control_already_treated": pl.Boolean,
                "estimate": pl.Float64,
                **{
                    name: pl.Int64
                    for name in [
                        "pre_start",
                        "pre_end",
                        "post_start",
                        "post_end",
                        "pre_periods",
                        "post_periods",
                    ]
                },
            },
        ),
        float(twfe),
        float(reconstructed),
        rr / (total * T),
        float(omega),
        float(beta_within) if beta_within is not None else None,
        float(within_contribution),
        metadata,
    )

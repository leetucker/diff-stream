"""Separate clean-sample BJS lead regression with bounded unit elimination.

The authors' pre-trend exercise augments the untreated unit/time regression
with lead indicators. It does not average residuals from the post-effect fit.
No author code is copied; the regression is implemented via sufficient sums.
"""

from copy import copy
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from hdfe_stream.core.results_base import RegressionResult

from ._core.nuisance import solve_checked


@dataclass(kw_only=True)
class BJSPlaceboResult(RegressionResult):
    """Pooled clean lead coefficients from a separate untreated-sample WLS fit.

    Standard coefficient/interval/Wald methods are inherited from the family.
    There is no estimated joint covariance with the main treatment effects.
    """

    n_units: int
    _support: pl.DataFrame = field(repr=False)
    estimator = "bjs_placebo"
    cells = ()
    _normal_inference = True

    def support(self):
        """Observed cohort/date support and original-scale weight totals."""
        return self._support.clone()

    def pretrend_test(self, *, level="auto", horizons=None):
        """Joint clean-lead Wald test; horizon selection does not refit the model."""
        from .pretrend import pretrend_test

        return pretrend_test(self, level=level, horizons=horizons)

    def with_vcov(self, vcov):
        if vcov not in ("CRV0", "CRV1"):
            raise ValueError("vcov must be 'CRV0' or 'CRV1'")
        factor = self.n_units / (self.n_units - 1)
        ratio = (factor if vcov == "CRV1" else 1) / (factor if self.vcov_type == "CRV1" else 1)
        out = copy(self)
        out.vcov = self.vcov * ratio
        out.vcov_type = vcov
        out.all_vcovs = {vcov: (out.vcov, out.df_t)}
        return out


def fit_placebos(store, pretrends, anticipation, *, fml, depvar, panel, vcov):
    """Eliminate unit intercepts locally, solve time/lead coefficients jointly."""
    if store.design is not None or not store.weights_time_invariant:
        from .linear import fit_placebos as fit_rich_placebos

        return fit_rich_placebos(
            store, pretrends, anticipation, fml=fml, depvar=depvar, panel=panel, vcov=vcov
        )
    T, L = store.periods, pretrends
    horizons = np.arange(-anticipation - L, -anticipation, dtype=np.int64)
    rows = []
    raw_scale = store.weight_scale / store.weight_normalizer
    support = np.zeros(L, dtype=int)
    for g, counts in sorted(store.actual_cell_counts.items()):
        if not g:
            continue
        for ti, t in enumerate(store.times):
            j = int(t) - g + anticipation + L
            if 0 <= j < L and counts[ti]:
                support[j] += counts[ti]
                rows.append(
                    dict(
                        aggregate=f"event::{horizons[j]}",
                        cohort=int(g),
                        time=int(t),
                        event_time=int(horizons[j]),
                        observed_units=int(counts[ti]),
                        weight_total=float(store.actual_cell_weights[g][ti] * raw_scale),
                    )
                )
    if np.any(support == 0):
        raise ValueError(
            f"BJS requested clean placebo horizons are not observed: {horizons[support == 0].tolist()}; reduce pretrends"
        )
    Z = np.eye(T)[:, 1:]
    P = T - 1 + L
    Q, q = np.zeros((P, P)), np.zeros(P)
    context = "BJS clean placebo design is unidentified (missing earlier clean reference or disconnected/collinear design); reduce pretrends"

    def batches():
        return store.batches(include_weights=True, actual_cohort=True)

    def design(g):
        lead = (store.times[:, None] - g == horizons) & (g != 0)
        return np.column_stack([Z, lead])

    def parts(g, values):
        obs = np.isfinite(values)
        use = obs & ((store.times < g - anticipation) if g else True)
        n = use.sum(axis=1)
        if np.any(n == 0):
            raise ValueError(
                "BJS requires at least one untreated observation per unit for clean placebo fitting"
            )
        return use, n, np.where(use, values, 0)

    n_obs = 0
    for g, y, w in batches():
        for cg in np.unique(g):
            mask = g == cg
            use, n, raw = parts(cg, y[mask])
            X, sw = design(cg), w[mask]
            xm = use @ X / n[:, None]
            centered = np.where(use, raw - raw.sum(axis=1)[:, None] / n[:, None], 0)
            Q += X.T @ ((sw @ use)[:, None] * X) - xm.T @ ((sw * n)[:, None] * xm)
            q += X.T @ (sw @ centered)
            n_obs += int(n.sum())
    coefficients = solve_checked(Q, q, context)
    H = solve_checked(Q, np.eye(P)[:, -L:], context)
    V = np.zeros((L, L))
    for g, y, w in batches():
        scores = np.zeros((len(g), L))
        for cg in np.unique(g):
            mask = g == cg
            use, n, raw = parts(cg, y[mask])
            X = design(cg)
            residual = np.where(use, raw - X @ coefficients, 0)
            residual = np.where(use, residual - residual.sum(axis=1)[:, None] / n[:, None], 0)
            scores[mask] = w[mask, None] * (residual @ X @ H)
        V += scores.T @ scores
    factor = store.n_units / (store.n_units - 1) if vcov == "CRV1" else 1.0
    V *= factor
    return BJSPlaceboResult(
        fml=fml,
        depvar=depvar,
        coefnames=[f"event::{h}" for h in horizons],
        beta=coefficients[-L:],
        vcov=(V + V.T) / 2,
        vcov_type=vcov,
        df_t=store.n_units - 1,
        all_vcovs={vcov: (V, store.n_units - 1)},
        n_obs=n_obs,
        n_clusters={panel[0]: store.n_units},
        n_units=store.n_units,
        weights=None,
        _support=pl.DataFrame(rows),
        diagnostics=dict(
            aggregation="dynamic",
            anticipation=anticipation,
            reference_event_time=None,
            fitted_horizons=horizons.tolist(),
            pretrends=L,
            base_period="all other clean observations; no fixed event-time reference",
            inference="separate clean-sample WLS; unit-clustered analytic scores",
            cohort_sizes={str(g): int(n) for g, n in store.actual_gc.items()},
            reference_observations=n_obs - int(support.sum()),
            normalized_future_cohorts="original timing retained for placebo indicators",
            post_effect_covariance="not estimated",
            warnings=[],
        ),
    )

"""Difference-in-differences with a continuous treatment (Callaway, Goodman-Bacon and Sant'Anna).

Each treated unit receives a dose when it is first treated and keeps it. For each
cohort and period, the outcome change of the cohort's units since the base period
is regressed on a B-spline in the dose, and the control group's mean change is
subtracted: ATT(d) is the fitted curve less that mean, ACRT(d) its derivative.

The fit reuses the Callaway-Sant'Anna cells, base periods and control groups
(``estimators.setup``), the cohort-period means of staging, and the cohort-block
covariance. The dose is staged as the only covariate. Three passes over the staged
units follow: the treated doses (for the knots), each cohort's spline moments, and
the units' influence scores.
"""

from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import pyarrow as pa
from hdfe_stream.core.profile import Component, Profile
from hdfe_stream.core.results_base import RegressionResult
from scipy.interpolate import BSpline
from scipy.special import ndtri
from stream_limits import MemoryBudget, current_rss_gb, detect_cpus, thread_scope

from . import _docs, _options
from ._core.dose_kernels import dose_scores
from ._core.multiplier import (
    MAX_DRAW_ENTRIES,
    StreamingMultiplier,
    multiplier_band,
    validate_bootstrap,
)
from ._core.partition import preprocess_budget_mb, staging_bytes
from ._present import present, public_diagnostics, public_profile
from .cohort_scores import CohortScoreCovariance
from .estimators import fit_means, setup
from .results import DIDResult
from .store import PanelStore, inspect

PUBLIC = "did_continuous"


class DoseBasis:
    """B-spline basis of the dose, with an intercept, as R's contdid builds it.

    The columns are an intercept and ``splines2::bSpline(d, degree, knots)``: the
    B-splines of the given degree on ``interior`` knots, without the first one.
    The boundary knots are the range of the treated doses; the same basis is used to
    fit the regressions and to evaluate the curves.
    """

    def __init__(self, interior, lower, upper, degree):
        self.interior = np.asarray(interior, dtype=float)
        self.lower, self.upper, self.degree = float(lower), float(upper), int(degree)
        knots = np.r_[[self.lower] * (degree + 1), self.interior, [self.upper] * (degree + 1)]
        size = len(knots) - degree - 1
        self._spline = BSpline(knots, np.eye(size), degree, extrapolate=True)
        self._slope = self._spline.derivative()
        self.width = size

    def _check(self, d):
        d = np.asarray(d, dtype=float)
        if np.any(d < self.lower) or np.any(d > self.upper):
            raise ValueError(
                f"doses must lie within the treated doses' range [{self.lower:g}, {self.upper:g}]"
            )
        return d

    def values(self, d):
        b = self._spline(self._check(d))
        b[:, 0] = 1.0
        return b

    def slopes(self, d):
        b = self._slope(self._check(d))
        b[:, 0] = 0.0
        return b


def _validated(name, value, low, kind=Integral):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, kind) or value < low:
        raise ValueError(f"{name} must be an integer >= {low}")
    return int(value)


def _outcome(fml):
    if not isinstance(fml, str) or fml.count("~") != 1:
        raise ValueError("use an outcome formula without covariates: 'y ~ 1'")
    y, rhs = [s.strip() for s in fml.split("~")]
    if rhs != "1" or not y.isidentifier():
        raise ValueError(
            "did_continuous takes no covariates or fixed effects: use 'y ~ 1' "
            "(the dose is given by dose=)"
        )
    return y


def overall_weights(cells, cohort_sizes):
    """contdid's overall weights: each cohort's share, split evenly over its post-treatment cells."""
    post = np.array([t >= g for g, t in cells])
    treated = sorted({g for g, t in cells})
    total = sum(cohort_sizes[g] for g in treated)
    w = np.zeros(len(cells))
    for g in treated:
        mine = post & np.array([c == g for c, t in cells])
        if mine.any():
            w[mine] = cohort_sizes[g] / total / mine.sum()
    if not np.isclose(w.sum(), 1.0):
        raise ValueError("no post-treatment cells to aggregate")
    return w


@dataclass(kw_only=True)
class DoseResponseResult(RegressionResult):
    """Dose-response effects of a continuous treatment.

    The coefficients are the overall ATT (the effect of being treated, at the
    doses received) and the overall ACRT (the average causal response, the
    derivative of ATT(d) averaged over the treated doses), as contdid reports them.
    ``dose_response()`` gives the curves ATT(d) and ACRT(d); ``aggregate()`` event
    studies and other aggregations of the binarized ATT or of the ACRT;
    ``cohort_time()`` the cohort-period estimates.
    """

    n_units: int
    _att: DIDResult = field(repr=False)
    _acrt: DIDResult = field(repr=False)
    _basis: DoseBasis = field(repr=False)
    _grid: np.ndarray = field(repr=False)
    _cells: list = field(repr=False)
    _weights: np.ndarray = field(repr=False)
    _theta: np.ndarray = field(repr=False)
    _theta_cov: np.ndarray = field(repr=False)
    _theta_draws: np.ndarray | None = field(default=None, repr=False)
    _internal: dict | None = field(default=None, repr=False)
    _periods: object = field(default=None, repr=False)
    estimator = "continuous"
    cells = ()
    _normal_inference = True

    def _target(self, target):
        if target not in ("level", "slope"):
            raise ValueError("target must be 'level' (ATT) or 'slope' (ACRT)")
        return self._att if target == "level" else self._acrt

    def _factor(self):
        return self.n_units / (self.n_units - 1) if self.vcov_type == "CRV1" else 1.0

    def _curve_map(self, target, dose):
        """The linear map from the cell parameters (coefficients, control means) to the curve."""
        M, K = len(self._cells), self._basis.width
        basis = self._basis.values(dose) if target == "level" else self._basis.slopes(dose)
        J = np.zeros((len(dose), M * K + M))
        for j, w in enumerate(self._weights):
            if w:
                J[:, j * K : (j + 1) * K] = w * basis
                if target == "level":
                    J[:, M * K + j] = -w
        return J

    def dose_response(self, target="level", dose=None, *, alpha=0.05, uniform=False):
        """ATT(d) (target="level") or ACRT(d) (target="slope") over doses.

        ``dose`` defaults to the grid of the fit (``dose_grid``). The curve averages
        the post-treatment cohort-period curves with the weights of the overall
        effects. Intervals are pointwise, from the analytic standard errors; with
        ``uniform=True`` they are simultaneous over the doses, from the multiplier
        bootstrap (fit with ``multiplier_bootstrap_reps``), studentized by the
        draws' interquartile range as R's contdid does.
        """
        self._target(target)
        if isinstance(alpha, bool) or not isinstance(alpha, Real) or not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        dose = self._grid if dose is None else np.atleast_1d(np.asarray(dose, dtype=float))
        J = self._curve_map(target, dose)
        estimate = J @ self._theta
        se = np.sqrt(
            np.maximum(np.einsum("ij,jk,ik->i", J, self._theta_cov, J), 0) * self._factor()
        )
        if uniform:
            if self._theta_draws is None:
                raise ValueError("fit with multiplier_bootstrap_reps >= 2 for uniform bands")
            draws = self._theta_draws @ J.T * np.sqrt(self._factor())
            band_se, critical = multiplier_band(draws, alpha, joint=True)
        else:
            band_se, critical = se, ndtri(1 - alpha / 2)
        return pl.DataFrame(
            {
                "dose": dose,
                "estimate": estimate,
                "std_error": se,
                "ci_low": estimate - critical * band_se,
                "ci_high": estimate + critical * band_se,
            }
        )

    def aggregate(self, kind="event_time", *, target="level", **options):
        """Aggregate the binarized ATT (target="level") or the ACRT (target="slope").

        ``kind`` and the options are those of ``DIDResult.aggregate``. The event study
        of the ACRT is contdid's ``aggregation="eventstudy", target_parameter="slope"``;
        that of the ATT is the Callaway-Sant'Anna event study of being treated.
        """
        return self._target(target).aggregate(kind, **options)

    def cohort_time(self, target="level"):
        """The cohort-period binarized ATT or ACRT, with standard errors."""
        return self._target(target).cohort_time()

    def with_vcov(self, vcov):
        from copy import copy

        if vcov not in ("CRV0", "CRV1"):
            raise ValueError("vcov must be 'CRV0' or 'CRV1'")
        factor = self.n_units / (self.n_units - 1)
        ratio = (factor if vcov == "CRV1" else 1) / (factor if self.vcov_type == "CRV1" else 1)
        out = copy(self)
        out.vcov = self.vcov * ratio
        out.vcov_type = vcov
        out.all_vcovs = {vcov: (out.vcov, out.df_t)}
        out._att, out._acrt = self._att.with_vcov(vcov), self._acrt.with_vcov(vcov)
        return out

    def _summary_fields(self):
        return {
            **super()._summary_fields(),
            "estimator": PUBLIC,
            "n_units": self.n_units,
            "diagnostics": self.diagnostics,
        }

    def _summary_lines(self):
        d = self.diagnostics
        return [
            f"estimator: {PUBLIC}; units: {self.n_units:,}",
            *super()._summary_lines(),
            f"dose spline: degree {d['degree']}, {d['interior_knots']} interior knots; "
            f"control group: {d['control_group']}",
        ]


def _continuous(
    fml,
    data,
    *,
    panel,
    dose,
    cohort=None,
    treatment=None,
    control_group="notyettreated",
    base_period="varying",
    anticipation=0,
    degree=3,
    interior_knots=0,
    dose_grid=None,
    vcov="CRV0",
    bootstrap_reps=0,
    bootstrap_seed=None,
    bootstrap_weights="rademacher",
    panel_mode="balanced",
    threads=None,
    memory_gb=None,
    memory_reserve=None,
    workdir=None,
    profile=False,
    batch_units=None,
    preprocess_mb=None,
):
    yname = _outcome(fml)
    if not isinstance(dose, str) or not dose.isidentifier():
        raise ValueError("dose must name a numeric column")
    if panel is None or len(panel) != 2 or dose in panel or dose == yname:
        raise ValueError("give panel=(unit, time); the dose must be a column of its own")
    degree = _validated("degree", degree, 1)
    interior_knots = _validated("interior_knots", interior_knots, 0)
    anticipation = _validated("anticipation", anticipation, 0)
    if control_group not in ("nevertreated", "notyettreated"):
        raise ValueError("unknown control_group")
    if base_period not in ("varying", "universal"):
        raise ValueError("unknown base_period")
    if vcov not in ("CRV0", "CRV1"):
        raise ValueError("vcov must be 'CRV0' or 'CRV1'")
    if panel_mode not in ("balanced", "complete"):
        raise ValueError(
            "did_continuous needs a balanced panel: incomplete_units must be 'error' or 'drop'"
        )
    if batch_units is not None:
        _validated("batch_units", batch_units, 1)
    if threads is not None:
        _validated("n_threads", threads, 1)
    validate_bootstrap(bootstrap_reps, bootstrap_seed, bootstrap_weights)
    bootstrap_reps = int(bootstrap_reps)

    budget = MemoryBudget(memory_gb, memory_reserve)
    preprocess_mb = preprocess_budget_mb(preprocess_mb, budget)
    baseline = current_rss_gb() or 0
    with thread_scope(threads):
        lf, _, design = inspect(
            f"{yname} ~ 0 + {dose}",  # the dose alone: no column of ones per row
            data,
            panel,
            cohort,
            treatment=treatment,
            allow_covariates=True,
        )
        if not budget.fits((preprocess_mb * 1e6 + 96e6) / 1e9):
            raise MemoryError(
                "preprocessing buffers exceed job headroom before staging; reduce preprocess_mb or request more memory"
            )
        if workdir is not None:
            Path(workdir).mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="diff-stream-", dir=workdir) as run:
            store = PanelStore(
                lf,
                run,
                batch_units or 16384,
                budget,
                preprocess_mb,
                design,
                panel_mode,
                anticipation,
            )
            if not store.x_time_invariant:
                raise ValueError("the dose must be constant within each unit")
            pa.default_memory_pool().release_unused()
            times, gc, N, means = store.times, store.gc, store.n_units, store.means
            groups, cells, bases, controls, sizes = setup(
                "cs", times, gc, control_group, base_period, anticipation
            )
            M, T, G = len(cells), len(times), len(groups)
            K = degree + interior_knots + 1
            W = M * (K + 3)
            treated_units = N - gc.get(0, 0)

            def buffers(B):
                numeric = 8 * (3 * W * W + 2 * B * W + 4 * B * (T + K) + G * K * (K + T))
                if bootstrap_reps:
                    numeric += 8 * (
                        2 * bootstrap_reps * W
                        + 2 * max(bootstrap_reps, min(B * bootstrap_reps, MAX_DRAW_ENTRIES))
                    )
                return numeric + 8 * treated_units

            B = min(batch_units or 16384, N)
            if batch_units is None:
                while B > 256 and not budget.fits(buffers(B) / 1e9 + 128e6 / 1e9):
                    B //= 2
            numeric = buffers(B)
            store.batch_units = B
            if profile:
                return Profile(
                    model=f"diff-stream {PUBLIC}",
                    threads=threads or detect_cpus(),
                    memory_limit_gb=budget.limit_gb,
                    memory_reserve=budget.reserve,
                    polars_threads=pl.thread_pool_size(),
                    baseline_gb=baseline,
                    components=[
                        Component(
                            "numerical buffers and runtime",
                            numeric + 192e6,
                            numeric * 1.5 + 256e6,
                        ),
                        Component(
                            "bounded partition preprocessing",
                            staging_bytes(preprocess_mb, store) + 96e6,
                            preprocess_mb * 1e6 + 256e6,
                        ),
                        Component(
                            "snapshot, hash partitions and sorted leaves",
                            72 * N * T,
                            144 * N * T,
                            where="disk",
                        ),
                    ],
                    notes={
                        "n_units": N,
                        "periods": T,
                        "cells": M,
                        "spline_columns": K,
                        "batch_units": B,
                        "partitions": store.partition_count,
                        "preprocess_mb": preprocess_mb,
                    },
                    warnings=[
                        "Heuristic estimate: source query, readers, writers and allocator retention can exceed allowances."
                    ],
                )
            if not budget.fits(numeric / 1e9 + 128e6 / 1e9):
                raise MemoryError(
                    "estimated numerical buffers exceed job headroom; reduce batch_units or request more memory"
                )

            # Pass 1: the treated units' doses, for the knots, the grid and the boundary.
            # One array of treated_units doses, filled in place; its quantiles are
            # taken in place too (np.quantile otherwise copies it).
            positive = np.empty(treated_units)
            filled = 0
            for g, _, x in store.batches(include_x=True):
                mine = x[g != 0, 0, 0]
                if np.any(mine <= 0):
                    raise ValueError("treated units must have a positive dose")
                positive[filled : filled + len(mine)] = mine
                filled += len(mine)
            lower, upper = positive.min(), positive.max()
            knot_levels = np.linspace(0, 1, interior_knots + 2)[1:-1]
            grid_levels = np.arange(10, 100) / 100 if dose_grid is None else np.array([])
            levels = np.quantile(positive, np.r_[knot_levels, grid_levels], overwrite_input=True)
            del positive
            interior = levels[:interior_knots]
            if np.any(np.diff(np.r_[lower, interior, upper]) <= 0):
                raise ValueError("the dose quantiles give repeated knots; use fewer interior_knots")
            basis = DoseBasis(interior, lower, upper, degree)
            if dose_grid is None:
                grid = levels[interior_knots:]
            else:
                grid = np.atleast_1d(np.asarray(dose_grid, dtype=float))
                if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all():
                    raise ValueError("dose_grid must be a one-dimensional array of doses")
                basis._check(grid)

            # Pass 2: each cohort's spline moments.
            A = np.zeros((G, K, K))
            S = np.zeros((G, K, T))
            slope_sums = np.zeros((G, K))
            for g, y, x in store.batches(include_x=True):
                treated = g != 0
                if not treated.any():
                    continue
                idx = np.searchsorted(groups, g[treated])
                d = x[treated, 0, 0]
                P, D, Y = basis.values(d), basis.slopes(d), y[treated]
                for k in np.unique(idx):
                    rows = idx == k
                    A[k] += P[rows].T @ P[rows]
                    S[k] += P[rows].T @ Y[rows]
                    slope_sums[k] += D[rows].sum(axis=0)
            ainv = np.zeros((G, K, K))
            for k, g in enumerate(groups):
                if g == 0:
                    continue
                if np.linalg.cond(A[k]) > 1e12:
                    raise ValueError(
                        f"the dose spline is not identified in cohort {g}: it has too few "
                        "distinct doses for the spline's columns; reduce degree or interior_knots"
                    )
                ainv[k] = np.linalg.inv(A[k])
            dbar = slope_sums / np.maximum(sizes, 1)[:, None]

            att, counts, args = fit_means(
                "cs", times, groups, cells, bases, controls, sizes, means, anticipation
            )
            gi, ti, _, _, nc, _, control_mean = args
            beta = np.array(
                [ainv[gi[j]] @ (S[gi[j]][:, ti[j]] - S[gi[j]][:, bases[j]]) for j in range(M)]
            ).reshape(M, K)
            acrt = np.einsum("jk,jk->j", dbar[gi], beta)

            # Pass 3: the units' influence scores.
            labels = np.r_[gi, gi, np.repeat(gi, K), gi]
            covariance = CohortScoreCovariance(labels)
            # A unit's scores are nonzero only in its cohort's cells (all four kinds)
            # and in the cells it is a control in (the ATT and the control mean).
            columns = {}
            for k in range(G):
                own, control = np.flatnonzero(gi == k), np.flatnonzero(controls[:, k])
                columns[k] = np.sort(
                    np.r_[
                        own,
                        control,
                        M + own,
                        (2 * M + K * own[:, None] + np.arange(K)).ravel(),
                        2 * M + M * K + control,
                    ]
                )
            # Each cohort's scores go to a compact layout of its own columns, so that
            # a batch sorted by cohort gives each cohort a contiguous block.
            W = M * (K + 3)
            local = np.zeros((G, W), dtype=np.int64)
            for k, used in columns.items():
                local[k, used] = np.arange(len(used))
            compact = max(len(used) for used in columns.values())
            # The multiplier bootstrap draws one multiplier per unit for all its
            # scores, so it takes them in one layout for every unit.
            identity = np.broadcast_to(np.arange(W), (G, W)).copy()
            multiplier = (
                StreamingMultiplier(M * K + M, bootstrap_reps, bootstrap_seed, bootstrap_weights)
                if bootstrap_reps
                else None
            )
            for g, y, x in store.batches(include_x=True):
                idx = np.searchsorted(groups, g)
                order = np.argsort(idx, kind="stable")
                idx, y, g, d = idx[order], y[order], g[order], x[order, 0, 0]
                treated = g != 0
                P, D = np.zeros((len(g), K)), np.zeros((len(g), K))
                if treated.any():
                    P[treated], D[treated] = basis.values(d[treated]), basis.slopes(d[treated])
                common = (gi, ti, bases, controls, nc, means, control_mean, sizes, beta, acrt)
                scores = dose_scores(idx, y, P, D, *common, ainv, dbar, local, compact)
                starts = np.r_[0, np.flatnonzero(idx[1:] != idx[:-1]) + 1, len(idx)]
                for a, b in zip(starts[:-1], starts[1:], strict=True):
                    k = int(idx[a])
                    used = columns[k]
                    covariance.add_block(k, scores[a:b, : len(used)], used, count=True)
                if multiplier is not None:
                    full = dose_scores(idx, y, P, D, *common, ainv, dbar, identity, W)
                    # in the staged unit order, so draws do not depend on the batches
                    multiplier.update(full[np.argsort(order), 2 * M :])
            V, cross, shares = covariance.blocks()

            factor = N / (N - 1) if vcov == "CRV1" else 1.0
            names = [f"g::{g}:t::{t}" for g, t in cells]
            weights = overall_weights(cells, gc)
            diagnostics = dict(
                control_group=control_group,
                base_period=base_period,
                anticipation=anticipation,
                degree=degree,
                interior_knots=interior_knots,
                knots=interior.tolist(),
                boundary_knots=[float(lower), float(upper)],
                cohort_sizes={str(g): int(n) for g, n in gc.items()},
                time_min=int(times[0]),
                time_max=int(times[-1]),
                input_units=store.n_units_input,
                dropped_units=store.n_units_input - N,
                periods=T,
                batch_units=B,
                partitions=store.partition_count,
                preprocess_mb=preprocess_mb,
                inference="unit clustered; analytic influence functions",
                warnings=[],
            )

            def cells_result(block, values):
                s = slice(block * M, (block + 1) * M)
                return present(
                    DIDResult(
                        fml=fml,
                        depvar=yname,
                        coefnames=list(names),
                        beta=values,
                        vcov=V[s, s] * factor,
                        vcov_type=vcov,
                        df_t=N - 1,
                        all_vcovs={vcov: (V[s, s] * factor, N - 1)},
                        n_obs=store.n_rows_used,
                        n_clusters={panel[0]: N},
                        estimator="cs",
                        cells=cells,
                        counts=counts,
                        observed_counts=np.array(
                            [store.cell_counts[g][t - times[0]] for g, t in cells], dtype=float
                        ),
                        cohort_counts=np.array([gc[g] for g, t in cells], dtype=float),
                        n_units=N,
                        diagnostics=dict(diagnostics),
                        _share_cross=cross[s, s] * factor,
                        _share_cov=(shares[s, s] - np.outer(counts, counts) / N) * factor,
                        _display=PUBLIC,
                    ),
                    store.period_labels,
                )

            att_result, acrt_result = cells_result(0, att), cells_result(1, acrt)
            # The overall ATT averages each cohort's post-treatment cells with the
            # cohorts' shares, which are estimated: as R did's and ptetools' "group"
            # aggregation, its influence function has a term for them, sum over
            # cohorts of the share's score times (the cohort's average - overall).
            # The overall ACRT is a fixed-weight average of the cells, as in contdid.
            joint = np.zeros((2, 2 * M))
            joint[0, :M], joint[1, M:] = weights, weights
            overall_vcov = joint @ V[: 2 * M, : 2 * M] @ joint.T
            n_cell = np.array([gc[g] for g, t in cells], dtype=float)
            share = weights * (att - weights @ att) / n_cell
            centered = shares[:M, :M] - np.outer(counts, counts) / N
            overall_vcov[0, 0] += 2 * weights @ cross[:M, :M] @ share + share @ centered @ share
            overall_vcov[0, 1] += share @ cross[M : 2 * M, M : 2 * M].T @ weights
            overall_vcov[1, 0] = overall_vcov[0, 1]
            overall_vcov *= factor
            theta = np.r_[beta.ravel(), control_mean]
            result = DoseResponseResult(
                fml=fml,
                depvar=yname,
                coefnames=["ATT", "ACRT"],
                beta=np.array([weights @ att, weights @ acrt]),
                vcov=overall_vcov,
                vcov_type=vcov,
                df_t=N - 1,
                all_vcovs={vcov: (overall_vcov, N - 1)},
                n_obs=store.n_rows_used,
                n_clusters={panel[0]: N},
                n_units=N,
                diagnostics=diagnostics,
                _att=att_result,
                _acrt=acrt_result,
                _basis=basis,
                _grid=grid,
                _cells=cells,
                _weights=weights,
                _theta=theta,
                _theta_cov=V[2 * M :, 2 * M :],
                _theta_draws=multiplier.draws if multiplier is not None else None,
                _periods=store.period_labels,
            )
            result._internal = dict(diagnostics)
            result.diagnostics = public_diagnostics(diagnostics, store.period_labels)
            return result


@_docs.document
def did_continuous(
    fml,
    data,
    *,
    panel,
    dose,
    cohort=None,
    treatment=None,
    control_group="not_yet_treated",
    base_period="varying",
    anticipation=0,
    degree=3,
    interior_knots=0,
    dose_grid=None,
    incomplete_units="error",
    vcov="CRV0",
    multiplier_bootstrap_reps=0,
    multiplier_bootstrap_seed=None,
    multiplier_bootstrap_weights="rademacher",
    n_threads=None,
    max_memory_gb=None,
    memory_reserve=None,
    workdir=None,
    profile=False,
    batch_units=None,
    preprocess_mb=None,
):
    """Callaway, Goodman-Bacon and Sant'Anna's difference-in-differences with a continuous treatment.

    Each treated unit receives a dose, constant over time, when it is first
    treated; untreated units have no dose. For each cohort and period, the outcome
    change of the cohort's units since the base period is regressed on a B-spline
    in the dose, and the control group's mean change is subtracted. That gives the
    average effect of each dose, ATT(d), and its derivative, the average causal
    response ACRT(d). The coefficients are their overall averages, as R's contdid
    reports them.

    Callaway, B., Goodman-Bacon, A. and Sant'Anna, P. H. C. (2024).
    Difference-in-differences with a continuous treatment. NBER Working Paper 32117.

    Parameters
    ----------
    fml : str
        ``"y ~ 1"``. Covariates are not supported.
    {DATA}
    dose : str
        Each unit's dose: positive and constant within a treated unit. The doses
        of never-treated units are not used.
    control_group : {"not_yet_treated", "never_treated"}, default "not_yet_treated"
        The comparison units: also units not yet treated by the period compared,
        or never-treated units only.
    base_period : {"varying", "universal"}, default "varying"
        The pre-treatment comparison period, as in ``did_callaway_santanna``.
    {ANTICIPATION}
    degree : int, default 3
        The degree of the dose spline (3: cubic).
    interior_knots : int, default 0
        Interior knots of the spline, at quantiles of the treated doses. With 0 the
        curve is a polynomial of the given degree.
    dose_grid : array, optional
        The doses at which ``dose_response()`` evaluates the curves. Default: the
        10th to 99th percentiles of the treated doses, in steps of 1, as in contdid.
    incomplete_units : {"error", "drop"}, default "error"
        What to do with units missing some periods: refuse the data or drop
        those units.
    {VCOV}
    multiplier_bootstrap_reps : int, default 0
        Draws of the multiplier bootstrap, for simultaneous confidence bands over
        doses (``dose_response(uniform=True)``). 0 skips it.
    multiplier_bootstrap_seed : int, optional
        Seed for the multiplier draws.
    multiplier_bootstrap_weights : {"rademacher", "mammen", "webb", "norm"}, default "rademacher"
        The distribution of the multiplier draws.
    {RESOURCES}

    Returns
    -------
    DoseResponseResult or Profile
        The overall ATT and ACRT; ``dose_response()`` the curves,
        ``aggregate()`` event studies of the binarized ATT or of the ACRT.
    """
    options = dict(locals())
    fml, data = options.pop("fml"), options.pop("data")
    return public_profile(_continuous(fml, data, **_options.internal(options)))

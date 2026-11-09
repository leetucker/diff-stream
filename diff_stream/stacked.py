"""Stacked difference-in-differences with corrective weights, without the stack.

Wing, Freedman and Hollingsworth (2024) stack one sub-experiment per adoption
cohort a: the cohort's units and its clean controls (never treated, or not
treated through the end of the event window), over event times lo..hi. The
weighted stacked regression

    y = a0 + a1 D + sum_e lambda_e 1[e] + sum_{e != r} delta_e D 1[e] + u,

with weight 1 for treated rows and (N^D_a / N^D) / (N^C_a / N^C) for the
controls of sub-experiment a, is saturated in (D, e), so each delta_h is a
difference of weighted cell means:

    delta_h = (m[1, h] - m[1, r]) - (m[0, h] - m[0, r]).

Its covariance clustered by unit is the sum over units of the squared
influence of those cell means, a unit's contributions summed over every
sub-experiment it enters. Both need two passes over whole units: the stacked
data, whose rows repeat each control once per sub-experiment it enters, are
never built.
"""

from copy import copy
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import pyarrow as pa
from hdfe_stream.core.influence import ClusterScoreAccumulator
from hdfe_stream.core.profile import Component, Profile
from stream_limits import MemoryBudget, current_rss_gb, detect_cpus, thread_scope

from . import _docs, _options
from ._core.partition import preprocess_budget_mb, staging_bytes
from ._present import label_frame, labels_of, present, public_profile
from .gardner import TwoStageResult
from .store import PanelStore, inspect


@dataclass(kw_only=True)
class StackedResult(TwoStageResult):
    """Event-time coefficients of the weighted stacked regression.

    ``select(horizons)`` reports a subset with its covariance, ``overall()``
    the average of the post-treatment coefficients, ``sub_experiments()`` the
    stack's composition and weights, and ``pretrend_test()`` tests the leads.
    """

    _sub_experiments: pl.DataFrame = None
    estimator = "stacked"

    def sub_experiments(self):
        """One row per sub-experiment: its cohort, units and corrective weights."""
        return label_frame(self._sub_experiments, labels_of(self))

    def overall(self, horizons=None):
        """The equally weighted average of post-treatment event-time effects.

        ``horizons`` selects the event times averaged (default: every fitted
        event time from 0 on).
        """
        events = [int(n.split("::")[1]) for n in self.coefnames]
        chosen = [e for e in events if e >= 0] if horizons is None else list(horizons)
        if not chosen or not set(chosen) <= set(events):
            raise ValueError("horizons must be fitted event times")
        a = np.zeros(len(events))
        for e in chosen:
            a[events.index(e)] = 1 / len(chosen)
        out = copy(self)
        out.coefnames = ["ATT"]
        out.beta = np.array([a @ self.beta])
        out.vcov = np.array([[a @ self.vcov @ a]])
        out.all_vcovs = {self.vcov_type: (out.vcov, self.df_t)}
        out._support = self._support.filter(
            pl.col("aggregate").is_in([f"event::{e}" for e in chosen])
        )
        out.diagnostics = {**self.diagnostics, "averaged_horizons": sorted(chosen)}
        return out


def _stack(store, window, anticipation):
    """The included sub-experiments, their clean controls and weights."""
    lo, hi = window
    first, last = int(store.times[0]), int(store.times[-1])
    groups = np.array(sorted(store.gw), dtype=np.int64)
    mass = np.array([store.gw[g] for g in groups])
    units = np.array([store.gc[g] for g in groups])
    rows, excluded = [], []
    for a in (int(g) for g in groups if g > 0):
        if a + lo < first or a + hi > last:
            excluded.append(dict(cohort=a, reason="event window outside the sample"))
            continue
        clean = (groups == 0) | (groups - anticipation > a + hi)
        if not clean.any():
            excluded.append(dict(cohort=a, reason="no unit untreated through the window"))
            continue
        rows.append(
            dict(
                cohort=a,
                n_treated_units=int(units[groups == a].sum()),
                n_control_units=int(units[clean].sum()),
                treated_weight_total=float(mass[groups == a].sum()),
                control_weight_total=float(mass[clean].sum()),
                control_cohorts=[int(g) for g in groups[clean]],
            )
        )
    if not rows:
        raise ValueError(
            "no sub-experiment fits: every cohort's event window leaves the sample or has "
            "no clean controls; narrow event_window"
        )
    table = pl.DataFrame(rows)
    nd, nc = table["treated_weight_total"].sum(), table["control_weight_total"].sum()
    table = table.with_columns(
        (pl.col("treated_weight_total") / nd).alias("treated_share"),
        ((pl.col("treated_weight_total") / nd) / (pl.col("control_weight_total") / nc)).alias(
            "control_weight"
        ),
    )
    return table, excluded


def _fit(store, table, window, anticipation):
    lo, hi = window
    ref = -anticipation - 1
    events = list(range(lo, hi + 1))
    L, t0 = len(events), int(store.times[0])
    r = events.index(ref)
    stack = [
        (a, np.array(c, dtype=np.int64), q)
        for a, c, q in zip(
            table["cohort"].to_list(),
            table["control_cohorts"].to_list(),
            table["control_weight"].to_list(),
            strict=True,
        )
    ]

    def windows(g, y, w):
        """Per sub-experiment: (treated rows, control rows, control weight, Y, W)."""
        for a, controls, q in stack:
            cols = slice(a + lo - t0, a + hi - t0 + 1)
            yield (g == a), np.isin(g, controls), q, y[:, cols], w[:, cols]

    sums, mass = np.zeros((2, L)), np.zeros((2, L))
    for g, y, w in store.batches(include_weights=True, row_weights=True):
        for treated, controls, q, Y, W in windows(g, y, w):
            observed = np.isfinite(Y)
            for d, rows, scale in ((1, treated, 1.0), (0, controls, q)):
                weight = np.where(observed[rows], W[rows], 0.0) * scale
                sums[d] += (weight * np.nan_to_num(Y[rows])).sum(axis=0)
                mass[d] += weight.sum(axis=0)
    if np.any(mass <= 0):
        missing = [events[j] for j in np.flatnonzero((mass <= 0).any(axis=0))]
        raise ValueError(f"no observations at event times {missing} in the stack")
    means = sums / mass
    keep = [j for j in range(L) if j != r]
    beta = (means[1, keep] - means[1, r]) - (means[0, keep] - means[0, r])
    accumulator = ClusterScoreAccumulator()
    for g, y, w in store.batches(include_weights=True, row_weights=True):
        score = np.zeros((len(g), L - 1))
        for treated, controls, q, Y, W in windows(g, y, w):
            observed = np.isfinite(Y)
            for d, rows, scale, sign in ((1, treated, 1.0, 1.0), (0, controls, q, -1.0)):
                if not rows.any():
                    continue
                weight = np.where(observed[rows], W[rows], 0.0) * scale
                influence = weight * (np.nan_to_num(Y[rows]) - means[d]) / mass[d]
                score[rows] += sign * (influence[:, keep] - influence[:, [r]])
        accumulator.add_clusters(score)
    names = [f"event::{events[j]}" for j in keep]
    support = pl.DataFrame(
        [
            dict(
                aggregate=f"event::{e}",
                cohort=a,
                time=a + e,
                event_time=e,
                treated_units=n,
            )
            for e in (events[j] for j in keep)
            for a, n in zip(table["cohort"].to_list(), table["n_treated_units"], strict=True)
        ]
    )
    return beta, accumulator.sums, names, support


def _stacked(
    fml,
    data,
    *,
    panel,
    event_window,
    cohort=None,
    treatment=None,
    anticipation=0,
    weights=None,
    allow_time_varying_weights=False,
    panel_mode="balanced",
    vcov="CRV0",
    threads=None,
    memory_gb=None,
    memory_reserve=None,
    workdir=None,
    profile=False,
    batch_units=4096,
    preprocess_mb=None,
):
    """Stacked DiD from internal option names; see did_stacked."""
    if (
        isinstance(anticipation, (bool, np.bool_))
        or not isinstance(anticipation, Integral)
        or anticipation < 0
    ):
        raise ValueError("anticipation must be a nonnegative integer")
    try:
        lo, hi = (int(e) for e in event_window)
    except (TypeError, ValueError) as exc:
        raise ValueError("event_window must be a pair of event times, (first, last)") from exc
    if any(isinstance(e, bool) or not isinstance(e, Integral) for e in event_window):
        raise ValueError("event_window must hold integer event times")
    if not lo <= -anticipation - 1 < 0 <= hi:
        raise ValueError(
            "event_window must include the reference period (-anticipation - 1) and event time 0"
        )
    if vcov not in ("CRV0", "CRV1"):
        raise ValueError("vcov must be 'CRV0' or 'CRV1'")
    if panel_mode not in ("balanced", "complete", "unbalanced"):
        raise ValueError("incomplete_units must be 'error', 'drop' or 'keep'")
    if threads is not None and (
        isinstance(threads, bool) or not isinstance(threads, int) or threads < 1
    ):
        raise ValueError("n_threads must be a positive integer")
    if isinstance(batch_units, bool) or not isinstance(batch_units, int) or batch_units < 1:
        raise ValueError("batch_units must be a positive integer")
    budget = MemoryBudget(memory_gb, memory_reserve)
    preprocess_mb = preprocess_budget_mb(preprocess_mb, budget)
    baseline = current_rss_gb() or 0
    with thread_scope(threads):
        lf, yname, design = inspect(fml, data, panel, cohort, treatment=treatment, weights=weights)
        if workdir is not None:
            Path(workdir).mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="diff-stream-stacked-", dir=workdir) as run:
            store = PanelStore(
                lf,
                run,
                batch_units,
                budget,
                preprocess_mb,
                None,
                panel_mode,
                anticipation,
                allow_time_varying_weights=allow_time_varying_weights,
                require_never_treated=False,
            )
            pa.default_memory_pool().release_unused()
            table, excluded = _stack(store, (lo, hi), anticipation)
            T, L, B = store.periods, hi - lo + 1, min(batch_units, store.n_units)
            numeric = 8 * (6 * L * L + 10 * B * T + 8 * B * L * len(table))
            diagnostics = dict(
                specification="dynamic",
                aggregation="event_time",
                anticipation=anticipation,
                reference_event_time=-anticipation - 1,
                event_window=[lo, hi],
                sub_experiments=table["cohort"].to_list(),
                excluded_cohorts=excluded,
                time_min=int(store.times[0]),
                time_max=int(store.times[-1]),
                panel_mode=panel_mode,
                balanced_sample=store.is_balanced,
                n_units_input=store.n_units_input,
                dropped_units=store.n_units_input - store.n_units,
                sampling_weights=weights,
                weight_convention="corrective weights (Wing, Freedman and Hollingsworth 2024)"
                + (" times sampling weights" if weights is not None else ""),
                partitions=store.partition_count,
                preprocess_mb=preprocess_mb,
                inference="unit clustered; units in several sub-experiments are one cluster",
            )
            if profile:
                return Profile(
                    model="diff-stream stacked",
                    threads=threads or detect_cpus(),
                    memory_limit_gb=budget.limit_gb,
                    memory_reserve=budget.reserve,
                    polars_threads=pl.thread_pool_size(),
                    baseline_gb=baseline,
                    components=[
                        Component(
                            "numerical buffers and runtime", numeric + 192e6, numeric * 1.5 + 256e6
                        ),
                        Component(
                            "bounded partition preprocessing",
                            staging_bytes(preprocess_mb, store) + 96e6,
                            preprocess_mb * 1e6 + 256e6,
                        ),
                        Component(
                            "sorted panel",
                            64 * store.n_units * T,
                            128 * store.n_units * T,
                            where="disk",
                        ),
                    ],
                    notes={**diagnostics, "n_units": store.n_units, "periods": T},
                    warnings=["Heuristic; the stacked rows are never built."],
                )
            if not budget.fits(numeric / 1e9 + 0.128):
                raise MemoryError(
                    "estimated numerical buffers exceed job headroom; reduce batch_units or request more memory"
                )
            beta, V, names, support = _fit(store, table, (lo, hi), anticipation)
            V *= store.n_units / (store.n_units - 1) if vcov == "CRV1" else 1
            V = (V + V.T) / 2
            result = StackedResult(
                fml=fml,
                depvar=yname,
                coefnames=names,
                beta=beta,
                vcov=V,
                vcov_type=vcov,
                df_t=store.n_units - 1,
                n_obs=store.n_rows_used,
                n_units=store.n_units,
                n_clusters={panel[0]: store.n_units},
                all_vcovs={vcov: (V, store.n_units - 1)},
                weights=weights,
                _support=support,
                _sub_experiments=table.drop("control_cohorts"),
                diagnostics={**diagnostics, "warnings": []},
            )
            return present(result, store.period_labels)


@_docs.document
def did_stacked(
    fml,
    data,
    *,
    panel,
    event_window,
    cohort=None,
    treatment=None,
    anticipation=0,
    weights=None,
    allow_time_varying_weights=False,
    incomplete_units="error",
    vcov="CRV0",
    n_threads=None,
    max_memory_gb=None,
    memory_reserve=None,
    workdir=None,
    profile=False,
    batch_units=4096,
    preprocess_mb=None,
):
    """Stacked difference-in-differences with corrective weights.

    One sub-experiment per adoption cohort: the cohort's units and the units
    not yet treated by the end of its event window (never-treated units and
    later cohorts), over that window. The sub-experiments are stacked and an
    event study is fitted on the stack, with the corrective weights of Wing,
    Freedman and Hollingsworth (2024), so that each event-time effect is the
    average of the sub-experiments' effects weighted by their treated units
    (their "trimmed aggregate ATT"). Only cohorts whose whole window lies in
    the sample are included. The stacked data, which repeat a control unit
    once for each sub-experiment it enters, are never built.

    Cengiz, D., Dube, A., Lindner, A. and Zipperer, B. (2019). The effect of
    minimum wages on low-wage jobs. *Quarterly Journal of Economics* 134(3),
    1405-1454.

    Wing, C., Freedman, S. M. and Hollingsworth, A. (2024). Stacked
    difference-in-differences. NBER Working Paper 32054.

    Parameters
    ----------
    fml : str
        ``"y ~ 1"``. Covariates are not supported.
    {DATA}
    event_window : tuple of int
        The first and last event times of each sub-experiment, e.g. ``(-4, 4)``;
        it must include the reference period (``-anticipation - 1``) and 0.
    {ANTICIPATION}
    {WEIGHTS}
    {INCOMPLETE_UNITS}
    {VCOV}
        A unit in several sub-experiments is one cluster.
    {RESOURCES}

    Returns
    -------
    StackedResult or Profile
        Event-time coefficients ``event::h`` (all but the reference period),
        with ``select()``, ``overall()``, ``pretrend_test()`` and
        ``sub_experiments()``.
    """
    options = dict(locals())
    fml, data = options.pop("fml"), options.pop("data")
    if fml.replace(" ", "").split("~")[1] != "1":
        raise ValueError("did_stacked takes no covariates: use a formula 'y ~ 1'")
    return public_profile(_stacked(fml, data, **_options.internal(options)))

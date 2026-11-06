"""DiD aggregation maps, cohort-support rules and composition tables.

Only cell/cohort-sized metadata is used. No source reread or unit-ID table is
needed to inspect contributions, even for a disk-backed BJS result.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import polars as pl

from ._present import internal


def integer(value):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def period_type(event_time, anticipation):
    if event_time >= 0:
        return "post_treatment"
    return "anticipation" if event_time >= -anticipation else "pre_treatment"


@dataclass
class AggregationPlan:
    kind: str
    names: list
    weights: np.ndarray
    contributors: pl.DataFrame
    composition: pl.DataFrame
    diagnostics: dict


def plan(result, kind, horizons, balanced_horizon, balance_window):
    if not result.cells:
        raise ValueError("aggregate the original group-time result")
    if kind not in ("event_time", "overall", "cohort", "calendar"):
        raise ValueError("kind must be 'event_time', 'overall', 'cohort' or 'calendar'")
    if balanced_horizon is not None and balance_window is not None:
        raise ValueError("choose balanced_horizon or balance_window, not both")
    if (balanced_horizon is not None or balance_window is not None) and kind != "event_time":
        raise ValueError("balancing options are available only for event-time aggregation")
    g = np.array([g for g, t in result.cells])
    t = np.array([t for g, t in result.cells])
    h = t - g
    anticipation = internal(result).get("anticipation", 0)
    cohorts = sorted(
        set(g.tolist()) | {int(k) for k in internal(result).get("cohort_sizes", {}) if int(k) != 0}
    )
    select = np.ones(len(h), dtype=bool)
    requested = None
    if horizons is not None:
        try:
            requested = list(horizons)
        except TypeError as exc:
            raise ValueError("horizons must be an iterable of integer event times") from exc
        if not all(integer(e) for e in requested):
            raise ValueError("horizons must contain integer event times")
        requested = sorted(set(int(e) for e in requested))
    last = internal(result).get("time_max", int(t.max()))
    first = internal(result).get(
        "time_min", last - internal(result).get("periods", int(t.max() - t.min() + 1)) + 1
    )
    span = last - first
    excluded = []
    if balanced_horizon is not None:
        if not integer(balanced_horizon) or balanced_horizon < 0:
            raise ValueError("balanced_horizon must be a nonnegative integer")
        balanced_horizon = int(balanced_horizon)
        if balanced_horizon > span:
            raise ValueError("no cohorts support the requested balanced window")
        required = set(range(balanced_horizon + 1))
        low, high = balanced_horizon - span, balanced_horizon
    elif balance_window is not None:
        try:
            bounds = list(balance_window)
        except TypeError as exc:
            raise ValueError("balance_window must be a pair of integer event times") from exc
        if len(bounds) != 2 or not all(integer(e) for e in bounds) or bounds[0] > bounds[1]:
            raise ValueError("balance_window must be an ordered pair of integer event times")
        low, high = (int(e) for e in bounds)
        balance_window = [low, high]
        if high - low > span:
            raise ValueError("no cohorts support the requested balanced window")
        if result.estimator == "bjs" and low < -anticipation:
            raise ValueError(
                "BJS balance_window must be post-treatment"
                if anticipation == 0
                else "BJS balance_window must start at event time >= -anticipation"
            )
        required = set(range(low, high + 1))
        # A normalized baseline is not an estimated cell or a support failure.
        if (
            result.estimator == "sunab"
            or (
                result.estimator == "etwfe"
                and internal(result).get("control_group") == "nevertreated"
            )
            or (result.estimator == "cs" and internal(result).get("base_period") == "universal")
        ):
            required.discard(-anticipation - 1)
    else:
        required = None
    if required is not None:
        keep = []
        for cg in cohorts:
            missing = sorted(required - set(h[g == cg].tolist()))
            if missing:
                excluded.append(
                    dict(
                        cohort=cg,
                        reason="missing required event-time support",
                        missing_horizons=missing,
                    )
                )
            else:
                keep.append(cg)
        select &= np.isin(g, keep) & (h >= low) & (h <= high)
    if requested is not None:
        select &= np.isin(h, requested)
    if kind != "event_time":
        select &= h >= 0
    if not select.any():
        raise ValueError("no supported cells in the requested aggregation")
    values = {"event_time": h, "cohort": g, "calendar": t}.get(kind)
    levels = sorted(set(values[select].tolist())) if values is not None else ["ATT"]
    prefix = {"event_time": "event", "cohort": "cohort", "calendar": "calendar"}.get(kind)
    names = ["ATT" if kind == "overall" else f"{prefix}::{v}" for v in levels]
    A = np.zeros((len(levels), len(h)))
    for j, level in enumerate(levels):
        mask = select if values is None else select & (values == level)
        A[j, mask] = result.counts[mask] / result.counts[mask].sum()
    contributing = sorted(set(g[select].tolist()))
    excluded_g = {d["cohort"] for d in excluded}
    excluded += [
        dict(cohort=cg, reason="no cells in requested aggregation", missing_horizons=[])
        for cg in cohorts
        if cg not in contributing and cg not in excluded_g
    ]
    observed = result.observed_counts
    populations = (
        result.cohort_counts
        if result.cohort_counts is not None
        else np.array(
            [
                internal(result).get("cohort_sizes", {}).get(str(cg), result.counts[j])
                for j, cg in enumerate(g)
            ]
        )
    )
    rows, summaries = [], []
    previous = None
    for j, name in enumerate(names):
        ix = np.flatnonzero(A[j])
        cg = sorted(set(g[ix].tolist()))
        weights = [float(A[j, g == k].sum()) for k in cg]
        current = dict(zip(cg, weights, strict=True))
        changed = previous is not None and set(current) != set(previous)
        weights_changed = previous is not None and (
            changed
            or any(not np.isclose(w, previous[k], atol=1e-12, rtol=0) for k, w in current.items())
        )
        summaries.append(
            dict(
                aggregate=name,
                n_cohorts=len(cg),
                n_cells=len(ix),
                cohorts=cg,
                cohort_weights=weights,
                cohort_units=int(sum(populations[next(k for k in ix if g[k] == c)] for c in cg)),
                observed_unit_periods=int(observed[ix].sum()) if observed is not None else None,
                target_weight_total=float(
                    result.counts[ix].sum() * internal(result).get("weight_scale", 1.0)
                ),
                cohorts_changed=changed,
                cohort_weights_changed=weights_changed,
            )
        )
        previous = current
        for k in ix:
            rows.append(
                dict(
                    aggregate=name,
                    cell=result.coefnames[k],
                    cohort=int(g[k]),
                    time=int(t[k]),
                    event_time=int(h[k]),
                    period_type=period_type(int(h[k]), anticipation),
                    estimate=float(result.beta[k]),
                    weight=float(A[j, k]),
                    weighted_estimate=float(A[j, k] * result.beta[k]),
                    weight_units=int(populations[k] if result.estimator == "cs" else observed[k])
                    if observed is not None
                    else int(result.counts[k]),
                    weight_total=float(
                        result.counts[k] * internal(result).get("weight_scale", 1.0)
                    ),
                    observed_units=int(observed[k]) if observed is not None else None,
                    cohort_units=int(populations[k]),
                )
            )
    composition = pl.DataFrame(summaries)
    diagnostics = dict(
        aggregation=kind,
        horizons=requested,
        balanced_horizon=balanced_horizon,
        balance_window=balance_window,
        contributing_cohorts=contributing,
        excluded_cohorts=excluded,
        cohort_composition_changes=bool(composition["cohorts_changed"].any()),
        cohort_weights_change=bool(composition["cohort_weights_changed"].any()),
        weight_basis=(
            "retained cohort sampling-weight totals"
            if result.estimator == "cs"
            else "observed cohort-time sampling-weight totals"
        )
        if internal(result).get("sampling_weights") is not None
        else "retained cohort units"
        if result.estimator == "cs"
        else "observed cohort-time units",
        observed_counts_available=result.observed_counts is not None,
    )
    return AggregationPlan(kind, names, A, pl.DataFrame(rows), composition, diagnostics)

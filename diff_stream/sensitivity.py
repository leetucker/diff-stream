"""Sensitivity of an event study to violations of parallel trends (Rambachan and
Roth, 2023), for the event studies of diff-stream's estimators.

The computation is hdfe_stream.core.sensitivity's, which takes an event study's
coefficients and covariance. This module finds them in a fit: the event-time
coefficients, their joint covariance, the omitted reference period and
anticipation, and refuses the fits whose pre-treatment coefficients do not
share a covariance with the effects (BJS) or are not cumulative relative to
one reference period (Callaway and Sant'Anna with a varying base period), as
R's HonestDiD does.
"""

import numpy as np
from hdfe_stream.core.plots import event_study_plot
from hdfe_stream.core.sensitivity import TrendSensitivityResult, event_study_sensitivity

from ._present import internal

__all__ = ["trend_sensitivity", "TrendSensitivityResult"]

SIGNATURE_DOC = """Sensitivity of the event study to violations of parallel trends
        (Rambachan and Roth, 2023; the method of R's HonestDiD).

        Confidence intervals for a post-treatment effect that remain valid
        when parallel trends fails by up to M, with M measured against the
        pre-treatment coefficients: by default (``"relative_magnitudes"``)
        no period-to-period change in the violation after treatment exceeds
        M times the largest one before it.

        Parameters
        ----------
        target : int, "average" or dict, optional
            The effect to bound: an event time (default 0, the first treated
            period), ``"average"`` over every event time after the reference,
            or weights ``{event time: weight}``.
        horizons : list of int, optional
            Event times to use (default: all estimated). They must be
            consecutive on each side of the reference period.
        restriction : {"relative_magnitudes", "smoothness", "smoothness_relative_magnitudes"}
            ``"smoothness"`` bounds the change in the violation's slope by M
            (M = 0 allows a linear trend); ``"smoothness_relative_magnitudes"``
            bounds it by M times the largest such change before treatment.
        M : float or list of float, optional
            Values of M (M-bar); by default R's grid for the restriction.
        bias : {"positive", "negative"}, optional
            Also restrict the sign of the violation after treatment.
        monotonicity : {"increasing", "decreasing"}, optional
            Also restrict the violation to be monotone.
        ci_method : {"fixed_length", "conditional", "hybrid_fixed_length", "hybrid_least_favorable"}, optional
            The interval (R's FLCI, Conditional, C-F, C-LF); R's default for
            the restriction.
        alpha : float, default 0.05
            One minus the coverage.
        **options
            ``grid_points`` (default 1000), ``grid_bounds``, ``simulations``
            (default 1000), ``seed`` (default 0), ``n_threads``.

        Returns
        -------
        TrendSensitivityResult
            ``.table`` (the robust interval at each M), ``.original``,
            ``.breakdown`` (the largest M at which the effect stays
            significant), ``.plot()`` (needs matplotlib)."""


def _event_coefficients(result, horizons, what):
    """(result holding the event study, event times, their indices, the reference
    event time or None) of a fit: the event-time aggregate of a cohort-time fit, the
    event-time coefficients of a two-stage or stacked fit, or did_dcdh's placebos and
    effects (placebo l at event time -l - 1, effect l at l - 1, reference -1)."""
    est = getattr(result, "estimator", None)
    if est == "dcdh":
        times, ix = [], []
        for j, name in enumerate(result.coefnames):
            kind, _, num = name.partition("::")
            if kind == "effect":
                times.append(int(num) - 1)
            elif kind == "placebo":
                times.append(-int(num) - 1)
            else:
                continue
            ix.append(j)
        reference = -1
    else:
        if getattr(result, "cells", None):
            result = result.aggregate("event_time")
        elif (
            est not in ("gardner", "stacked")
            and internal(result).get("aggregation") != "event_time"
        ):
            raise ValueError(
                f"{what} needs cohort-time estimates or an event-time aggregate "
                "(aggregate('event_time'))"
            )
        reference = internal(result).get("reference_event_time")
        reference = None if reference is None else int(reference)
        times, ix = [], []
        for j, name in enumerate(result.coefnames):
            kind, _, num = name.rpartition("::")
            if kind.endswith("event"):
                times.append(int(num))
                ix.append(j)
        if not ix:
            raise ValueError(
                "no event-time coefficients in this fit; fit an event study "
                "(did_two_stage with event_study=True, say)"
            )
    if horizons is not None:
        # the omitted reference may be named (horizons=range(-3, 3), say)
        keep = {int(h) for h in horizons} - {reference}
        missing = keep - set(times)
        if missing:
            raise ValueError(f"horizons {sorted(missing)} are not estimated event times")
        sel = [k for k, t in enumerate(times) if t in keep]
        times = [times[k] for k in sel]
        ix = [ix[k] for k in sel]
    return result, times, ix, reference


def _event_study(result, horizons):
    """(event times, beta, vcov, reference) of a fit's event study, for the
    sensitivity analysis."""
    est = getattr(result, "estimator", None)
    if est in ("bjs", "bjs_placebo"):
        raise ValueError(
            "trend_sensitivity is not available for BJS: its pre-treatment leads are a separate "
            "regression (pretrend_leads=) with no covariance with the treatment effects, and the "
            "sensitivity analysis needs both in one covariance matrix. Use did_callaway_santanna "
            "(base_period='universal'), did_sun_abraham, did_two_stage or did_stacked"
        )
    if est == "cs" and internal(result).get("base_period") != "universal":
        raise ValueError(
            "trend_sensitivity needs base_period='universal' for Callaway and Sant'Anna: "
            "with a varying base period the pre-treatment coefficients are period-to-period "
            "changes, not cumulative differences from one reference period (R's honest_did "
            "refuses them too). Refit with did_callaway_santanna(..., base_period='universal')"
        )
    res, times, ix, reference = _event_coefficients(result, horizons, "trend_sensitivity")
    if est == "dcdh" and not any(t < -1 for t in times):
        raise ValueError("fit did_dcdh with placebos >= 1 for a sensitivity analysis")
    if reference is None:
        raise ValueError(
            "this fit has no event-study reference period (a static two-stage fit, say); "
            "fit an event study with an omitted reference period"
        )
    if not any(t < reference for t in times):
        raise ValueError(
            f"no pre-treatment coefficients (event times before the reference {reference}): "
            f"this fit estimates only {sorted(times)}. The sensitivity analysis bounds the "
            "violation after treatment by the one before it, so it needs at least one "
            "pre-treatment coefficient"
        )
    return times, res.beta[ix], res.vcov[np.ix_(ix, ix)], reference


def plot_event_study(
    result, *, horizons=None, alpha=0.05, joint=False, ax=None, title=None, **joint_options
):
    res, times, ix, reference = _event_coefficients(result, horizons, "plot_event_study")
    names = [res.coefnames[i] for i in ix]
    ci = res.confint(alpha=alpha, keep=names, exact_match=True, joint=joint, **joint_options)
    level = f"{(1 - alpha) * 100:g}%{' simultaneous' if joint else ''} interval"
    return event_study_plot(
        times,
        res.beta[ix],
        ci[:, 1].to_numpy(),
        ci[:, 2].to_numpy(),
        reference=reference,
        ax=ax,
        title=title,
        ylabel=f"Estimate and {level}",
    )


PLOT_DOC = """Plot the event study: each event time's estimate and confidence interval,
        the omitted reference period at zero (hollow) and a dashed line after it, as
        pyfixest's iplot does. Cohort-time fits are aggregated by event time first;
        did_dcdh plots placebo l at event time -l - 1 and effect l at l - 1. Needs
        matplotlib, which is not a dependency (``pip install diff-stream[plot]``).

        Parameters
        ----------
        horizons : list of int, optional
            Event times to plot (default: all estimated).
        alpha : float, default 0.05
            One minus the coverage.
        joint : bool, default False
            Simultaneous intervals over the plotted event times (``confint(joint=True)``;
            ``reps`` and ``seed`` pass through) instead of pointwise ones.
        ax : matplotlib.axes.Axes, optional
            Draw here instead of on a new figure.
        title : str, optional

        Returns
        -------
        matplotlib.figure.Figure"""


def trend_sensitivity(
    result,
    *,
    target=None,
    horizons=None,
    restriction="relative_magnitudes",
    M=None,
    bias=None,
    monotonicity=None,
    ci_method=None,
    alpha=0.05,
    **options,
):
    times, beta, vcov, reference = _event_study(result, horizons)
    return event_study_sensitivity(
        times,
        beta,
        vcov,
        reference=reference,
        target=target,
        restriction=restriction,
        M=M,
        bias=bias,
        monotonicity=monotonicity,
        ci_method=ci_method,
        alpha=alpha,
        **options,
    )


trend_sensitivity.__doc__ = (
    "Sensitivity of a fit's event study to violations of parallel trends; see\n"
    "DIDResult.trend_sensitivity.\n\n" + SIGNATURE_DOC
)

"""The refit bootstrap of diff-stream's estimators: resample units (or clusters) and refit.

The resampling, the refits in worker processes, the failure ledger and the
intervals are hdfe_stream.core's (``core.refit``). This module owns what is
particular to the estimators: which columns a fit needs, the sample and comparison
group the original fit chose (held fixed across draws), the checks that a draw
keeps them, and the dispatch to each estimator.
"""

import warnings
from inspect import signature
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
from hdfe_stream.core.partition import preprocess_budget_mb
from hdfe_stream.core.refit import (
    BootstrapFailureWarning,
    RefitBootstrapResult,
    refit_draws,
    refit_workers,
)
from stream_limits import MemoryBudget, thread_scope

from . import _options
from .api import _fit, did_bjs, did_callaway_santanna, did_sun_abraham
from .dcdh import did_dcdh
from .etwfe import did_extended_twfe
from .gardner import did_two_stage
from .store import inspect, source

__all__ = ["BootstrapFailureWarning", "RefitBootstrapResult", "refit_bootstrap"]


def _target(result, aggregate, aggregate_options, coefficients):
    target = result.aggregate(aggregate, **aggregate_options) if aggregate is not None else result
    names = list(target.coefnames) if coefficients is None else list(coefficients)
    if not names or len(set(names)) != len(names) or not set(names) <= set(target.coefnames):
        raise ValueError("requested coefficient names are missing, duplicated or empty")
    values = target.beta[[target.coefnames.index(n) for n in names]].copy()
    if not np.isfinite(values).all():
        raise ValueError("requested coefficients are nonfinite or suppressed")
    return names, values


def _replicate_formula(fml, unit):
    if "|" not in fml:
        return fml
    regression, fe = fml.split("|")
    terms = [x.strip() for x in fe.split("+") if x.strip() != unit]
    return regression if not terms else regression + "|" + "+".join(terms)


def _estimate(fml, path, *, panel, cohort, treatment, estimator, workdir, options):
    """One fit of an estimator, by its internal code and option names."""
    if estimator == "dcdh":
        from .dcdh import _dcdh  # noqa: PLC0415 (dcdh's internal function)

        return _dcdh(fml, path, panel=panel, treatment=treatment, workdir=workdir, **options)
    return _fit(
        fml,
        path,
        panel=panel,
        cohort=cohort,
        treatment=treatment,
        estimator=estimator,
        workdir=workdir,
        **options,
    )


def _refit_task(task):
    """One draw's refit (core.refit calls it, in this process or a worker)."""
    if task["dates"] != task["expected_dates"]:
        raise ValueError("resample loses dates from the fixed analysis window")
    options = dict(task["options"], threads=task["threads"], memory_gb=task["memory_gb"])
    result = None
    try:
        result = _estimate(
            task["fml"],
            task["path"],
            panel=(task["unit_column"], task["time"]),
            cohort=task["cohort"],
            treatment=task["treatment"],
            estimator=task["estimator"],
            workdir=task["workdir"],
            options=options,
        )
        # Resampled units keep the original sample; resampled clusters vary in size.
        if task["n_units"] is not None and (
            result.n_units != task["n_units"] or result.n_obs != task["rows"]
        ):
            raise ValueError("refit changed the fixed sampled population")
        _, values = _target(result, task["aggregate"], task["aggregate_options"], task["names"])
        return values
    finally:
        if result is not None:
            result.cleanup()


def _dcdh_columns(fml, panel, treatment, options):
    """The source columns a did_dcdh fit reads."""
    import formulaic

    y, rhs = [s.strip() for s in fml.split("~")]
    covariates = [] if rhs == "1" else sorted(formulaic.Formula(rhs).required_variables)
    trends = options.get("trends_by")
    trends = [] if trends is None else ([trends] if isinstance(trends, str) else list(trends))
    extra = [options.get("weights"), options.get("cluster")]
    return [*panel, treatment, y, *covariates, *trends, *[c for c in extra if c is not None]]


def _refit_bootstrap(
    fml,
    data,
    *,
    panel,
    cohort=None,
    treatment=None,
    estimator,
    public_estimator,
    public_options,
    reps=199,
    seed=None,
    aggregate=None,
    aggregate_options=None,
    coefficients=None,
    workers=1,
    threads=None,
    memory_gb=None,
    memory_reserve=None,
    preprocess_mb=None,
    batch_units=256,
    workdir=None,
    **fit_options,
):
    """refit_bootstrap with internal estimator codes and option names."""
    if isinstance(batch_units, bool) or not isinstance(batch_units, int) or batch_units < 1:
        raise ValueError("batch_units must be an integer >= 1")
    refit_workers(reps, workers, threads)
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    dcdh = estimator == "dcdh"
    if aggregate not in (None, "event_time", "overall", "cohort", "calendar"):
        raise ValueError("aggregate must be 'event_time', 'overall', 'cohort', 'calendar' or None")
    if estimator in ("gardner", "dcdh") and aggregate is not None:
        raise ValueError(
            f"{public_estimator} coefficients are reported directly: use aggregate=None"
            + (" (and event_study=False for a single ATT)" if estimator == "gardner" else "")
        )
    aggregate_options = {} if aggregate_options is None else dict(aggregate_options)
    if aggregate is None and aggregate_options:
        raise ValueError("aggregate_options requires an aggregate target")
    if isinstance(coefficients, str):
        raise ValueError("coefficients must be a sequence of exact coefficient names")
    if coefficients is not None:
        coefficients = list(coefficients)
    if (
        fit_options.get("profile", False)
        or fit_options.get("bootstrap_reps", 0)
        or fit_options.get("bootstrap_seed") is not None
    ):
        raise ValueError(
            "refit_bootstrap fits the model itself: profile and the multiplier bootstrap "
            "options cannot be passed to it"
        )
    budget = MemoryBudget(memory_gb, memory_reserve)
    # Refits size their own buffers from their own budgets (workers have less).
    # The draw plan follows the sampler's unit order, which depends on its
    # partitions: a fixed default keeps seeded draws the same on every machine.
    refit_preprocess_mb = preprocess_mb
    preprocess_mb = preprocess_budget_mb(64 if preprocess_mb is None else preprocess_mb, budget)
    if not budget.fits((preprocess_mb * 1e6 + 96e6) / 1e9):
        raise MemoryError("bootstrap preprocessing buffers exceed job headroom")
    cluster = fit_options.get("cluster") if dcdh else None
    if cluster == panel[0]:
        cluster = None
    if dcdh:
        required = list(dict.fromkeys(_dcdh_columns(fml, panel, treatment, fit_options)))
    else:
        weights = fit_options.get("weights")
        _, y, design, absorbed, _ = inspect(
            fml,
            data,
            panel,
            cohort,
            treatment=treatment,
            allow_covariates=True,
            weights=weights,
            allow_fe=estimator in ("sunab", "bjs", "gardner"),
            fe_solver=fit_options.get("fe_solver", "auto"),
        )
        required = list(
            dict.fromkeys(
                [
                    *panel,
                    cohort if cohort is not None else treatment,
                    y,
                    *([] if weights is None else [weights]),
                    *([] if design is None else design.required),
                    *[c for _, written in absorbed or [] for c in written.split("^")],
                ]
            )
        )
    src = source(data)
    missing = set(required) - set(src.collect_schema().names())
    if missing:
        raise ValueError(f"unknown columns: {sorted(missing)}")
    options = dict(
        fit_options,
        threads=threads,
        memory_gb=memory_gb,
        memory_reserve=memory_reserve,
        preprocess_mb=refit_preprocess_mb,
        batch_units=batch_units,
    )
    if workdir is not None:
        Path(workdir).mkdir(parents=True, exist_ok=True)
    with thread_scope(threads), TemporaryDirectory(prefix="diff-stream-refit-", dir=workdir) as run:
        run = Path(run)
        snapshot = run / "source.parquet"
        src.select(required).sink_parquet(snapshot, engine="streaming", row_group_size=32768)
        original_dates = (
            pl.scan_parquet(snapshot)
            .select(
                pl.col(panel[1]).min().alias("lo"),
                pl.col(panel[1]).max().alias("hi"),
                pl.len().alias("rows"),
            )
            .collect(engine="streaming")
            .row(0)
        )
        original = _estimate(
            fml,
            snapshot,
            panel=panel,
            cohort=cohort,
            treatment=treatment,
            estimator=estimator,
            workdir=run / "fit",
            options=options,
        )
        try:
            names, beta = _target(original, aggregate, aggregate_options, coefficients)
            n_units, n_obs = original.n_units, original.n_obs
            diag = dict(original.diagnostics)
        finally:
            original.cleanup()
        del original
        if dcdh:
            periods = diag["periods"]
            step = (diag["time_max"] - diag["time_min"]) // max(periods - 1, 1)
        else:
            step = diag.get("period_step", 1)
        expected_dates = list(range(diag["time_min"], diag["time_max"] + 1, step))
        cutoff = diag.get("reference_cutoff")
        mode = options.get("panel_mode", "balanced")

        def select(frame):
            if mode == "complete":
                ids = frame[panel[0]].to_numpy()
                starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
                lengths = np.diff(np.r_[starts, len(frame)])
                frame = frame.filter(
                    pl.Series(
                        np.repeat(
                            lengths == (original_dates[1] - original_dates[0]) // step + 1, lengths
                        )
                    )
                )
            if cutoff is not None:
                frame = frame.filter(pl.col(panel[1]) < cutoff)
            return frame

        options.update(panel_mode="unbalanced" if mode == "unbalanced" else "balanced")
        if dcdh:
            # the estimates do not depend on the clusters, which the draws resample
            options.pop("cluster", None)
        else:
            options.update(reference_group="nevertreated")
        task = dict(
            fml=_replicate_formula(fml, panel[0]),
            time=panel[1],
            cohort=cohort,
            treatment=treatment,
            estimator=estimator,
            options=options,
            expected_dates=expected_dates,
            n_units=None if cluster is not None else n_units,
            aggregate=aggregate,
            aggregate_options=aggregate_options,
            names=names,
        )
        out = refit_draws(
            snapshot,
            run,
            fit=_refit_task,
            task=task,
            width=len(names),
            reps=reps,
            seed=seed,
            workers=workers,
            threads=threads,
            memory_gb=memory_gb,
            budget=budget,
            preprocess_mb=preprocess_mb,
            key=cluster if cluster is not None else panel[0],
            within=panel[0] if cluster is not None else None,
            sort_columns=[cluster, *panel] if cluster is not None else list(panel),
            select=select,
            time_column=panel[1],
            expected_groups=n_units,
            expected_rows=n_obs,
            worker_minimum_gb=0.35 + (refit_preprocess_mb or 64) / 1000 + 0.096,
            preload=("diff_stream",),
        )
        diagnostics = dict(
            out.diagnostics,
            unit_count=n_units,
            original_rows=n_obs,
            analysis_rows=n_obs,
            source_rows=original_dates[2],
            source_panel=list(panel),
            cohort=cohort,
            fit_options=public_options,
            aggregate=aggregate,
            aggregate_options=aggregate_options,
            incomplete_units={"balanced": "error", "complete": "drop", "unbalanced": "keep"}[mode],
            fixed_reference=diag.get("reference"),
            fixed_reference_cohort=diag.get("reference_cohort"),
            reference_cutoff=cutoff,
            analysis_time_min=diag["time_min"],
            analysis_time_max=diag["time_max"],
            replicate_reference=None if dcdh else "never_treated",
            sampling=(
                f"ordinary uniform clusters of {cluster} with replacement; fresh group IDs in "
                "each copy"
                if cluster is not None
                else "ordinary uniform unit pairs with replacement; fresh ID for each copy"
            ),
            inference="pointwise; complete successful rows only; failed attempts never redrawn",
            source_columns=required,
            batch_units=batch_units,
        )
    result = RefitBootstrapResult(
        fml=fml,
        estimator=public_estimator,
        coefnames=names,
        beta=beta,
        diagnostics=diagnostics,
        _draws=out.draws,
        _failures=out.failures,
        _warnings=out.warnings,
    )
    if out.failures:
        warnings.warn(
            f"{len(out.failures)} of {reps} refits failed. Inspect failures(); incomplete-run "
            "inference requires allow_incomplete=True and is conditional on successful fits.",
            BootstrapFailureWarning,
            stacklevel=2,
        )
    return result


# refit_bootstrap() sets these itself, for the original fit and every refit.
_CONTROLLED = {
    "fml",
    "data",
    "panel",
    "cohort",
    "treatment",
    "n_threads",
    "max_memory_gb",
    "memory_reserve",
    "workdir",
    "profile",
    "batch_units",
    "preprocess_mb",
    "multiplier_bootstrap_reps",
    "multiplier_bootstrap_seed",
    "multiplier_bootstrap_weights",
}


def refit_bootstrap(
    fml,
    data,
    *,
    panel,
    cohort=None,
    treatment=None,
    estimator,
    reps=199,
    seed=None,
    aggregate=None,
    aggregate_options=None,
    coefficients=None,
    n_workers=1,
    n_threads=None,
    max_memory_gb=None,
    memory_reserve=None,
    workdir=None,
    batch_units=256,
    preprocess_mb=None,
    **fit_options,
):
    """Bootstrap by resampling units (or clusters) and refitting the estimator.

    Draws ``reps`` samples of the units with replacement (each copy of a unit
    gets its own ID) and refits the whole estimator on each, including its
    propensity scores, trimming and aggregation weights. For ``did_dcdh`` with
    ``cluster=``, whole clusters are drawn instead, as did_multiplegt_dyn's
    bootstrap does, each copy's groups with IDs of their own. Reports bootstrap
    standard errors and percentile, basic or normal intervals for each
    coefficient. The sample and comparison group chosen by the original fit
    (incomplete units dropped, a last-treated reference window) are held
    fixed across draws. This is slower than analytic standard errors, by
    about ``reps`` fits; for CS, the multiplier bootstrap
    (``multiplier_bootstrap_reps``) is far cheaper.

    Parameters
    ----------
    fml, data, panel, cohort, treatment
        As for the estimator (``did_dcdh`` takes ``treatment``).
    estimator : function
        One of the estimator functions: ``did_callaway_santanna``,
        ``did_sun_abraham``, ``did_bjs``, ``did_two_stage``, ``did_extended_twfe``
        or ``did_dcdh``.
    reps : int, default 199
        Bootstrap draws.
    seed : int, optional
        Seed for the draws. With the same data layout, the same seed gives
        the same draws.
    aggregate : {"event_time", "overall", "cohort", "calendar"}, optional
        Bootstrap an aggregate (``result.aggregate(aggregate,
        **aggregate_options)``) instead of the fitted coefficients. Not
        available for ``did_two_stage`` and ``did_dcdh``, whose coefficients
        are reported directly.
    aggregate_options : dict, optional
        Options for ``aggregate()``, such as ``{"balanced_horizon": 2}``.
    coefficients : list of str, optional
        Bootstrap only these coefficients (exact names).
    n_workers : int, default 1
        Refits run in parallel, each in a fresh process. ``n_threads`` and
        ``max_memory_gb`` are shared out among them.
    n_threads, max_memory_gb, memory_reserve, workdir, batch_units, preprocess_mb
        As for the estimator; they cover the whole job.
    **fit_options
        The estimator's other options, e.g. ``adjustment="ipw"``.

    Returns
    -------
    RefitBootstrapResult
        ``tidy()``, ``confint()`` and ``standard_errors()``. Failed draws are
        recorded in ``failures()``, never redrawn; inference then requires
        ``allow_incomplete=True``.
    """
    code = _ESTIMATOR_CODES.get(estimator)
    if code is None:
        raise TypeError(
            "estimator must be one of diff-stream's estimator functions, such as did_bjs"
        )
    allowed = set(signature(estimator).parameters) - _CONTROLLED
    unknown = sorted(set(fit_options) - allowed)
    if any(name.startswith("multiplier_bootstrap") or name == "profile" for name in unknown):
        raise ValueError(
            "refit_bootstrap fits the model itself: profile and the multiplier bootstrap "
            "options cannot be passed to it"
        )
    if unknown:
        raise TypeError(f"{estimator.__name__} has no option {', '.join(unknown)}")
    return _refit_bootstrap(
        fml,
        data,
        panel=panel,
        cohort=cohort,
        treatment=treatment,
        estimator=code,
        public_estimator=estimator.__name__,
        public_options=dict(fit_options),
        reps=reps,
        seed=seed,
        aggregate=aggregate,
        aggregate_options=aggregate_options,
        coefficients=coefficients,
        workers=n_workers,
        threads=n_threads,
        memory_gb=max_memory_gb,
        memory_reserve=memory_reserve,
        workdir=workdir,
        batch_units=batch_units,
        preprocess_mb=preprocess_mb,
        **_options.internal(fit_options),
    )


_ESTIMATOR_CODES = {
    did_dcdh: "dcdh",
    did_callaway_santanna: "cs",
    did_sun_abraham: "sunab",
    did_bjs: "bjs",
    did_two_stage: "gardner",
    did_extended_twfe: "etwfe",
}

"""Public estimation entry points."""

from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import pyarrow as pa
from hdfe_stream.core.profile import Component, Profile
from stream_limits import MemoryBudget, current_rss_gb, detect_cpus, thread_scope

from ._core.multiplier import MAX_DRAW_ENTRIES, StreamingMultiplier, validate_bootstrap
from .bjs import BJSInference
from .bjs_placebo import fit_placebos
from .cs import fit_cs
from .effective_sample import finish_result, threshold
from .estimators import change_scores, fit_means, setup
from .results import DIDResult
from .store import PanelStore, inspect


def diff_stream(
    fml,
    data,
    *,
    panel,
    cohort,
    estimator="cs",
    control_group="nevertreated",
    base_period="varying",
    vcov="CRV0",
    batch_units=4096,
    threads=None,
    memory_gb=None,
    workdir=None,
    profile=False,
    preprocess_mb=64,
    avgeffectsby="cohort_time",
    leaveout=False,
    est_method="dr",
    trim_level=0.995,
    panel_mode="balanced",
    anticipation=0,
    weights=None,
    allow_time_varying_weights=False,
    bootstrap_reps=0,
    bootstrap_seed=None,
    bootstrap_weights="rademacher",
    pretrends=0,
    ess_warn_threshold=30,
    ess_suppress_threshold=None,
    specification="dynamic",
    reference_period=None,
    fixed_effects="cohort",
    inference="unconditional",
    max_design_columns=2048,
    reference_group="auto",
    balance=False,
    balance_ecdf=False,
):
    """Fit staggered difference-in-differences on an absorbing-treatment panel.

    Parameters
    ----------
    fml : str
        Column-name outcome formula. CS supports covariates, transformations,
        interactions and categorical coding (up to 256 design columns), e.g.
        ``"y ~ x + C(region)"``. BJS/SA also accept covariates and additional
        fixed effects, e.g. ``"y ~ x | region^time"``. Unit/time intercepts
        are always included; the compiled additional design is limited to 256
        columns. Unit-interacted effects and varying slopes are unsupported.
    data : polars.DataFrame, polars.LazyFrame or path
        Source frame or Parquet path/glob. Lazy execution explicitly uses the
        streaming engine. A snapshot is staged before row validation.
    panel : tuple[str, str]
        Unit and time columns; time is consecutive positive integers.
    cohort : str
        Constant actual first treatment period per unit. Zero denotes never treated;
        future cohorts are never-treated only if g-anticipation exceeds the sample. Always-treated units are unsupported.
    estimator : {"cs", "sunab", "bjs", "gardner", "etwfe"}
        Callaway–Sant'Anna, Sun–Abraham, BJS imputation, or Gardner two-stage DiD.
        Gardner returns direct static/dynamic coefficients; see gardner_stream.
        ETWFE fits saturated linear interactions; see etwfe_stream.
    panel_mode : {"balanced", "complete", "unbalanced"}
        Default "balanced" rejects missing unit-time rows. "complete" drops
        incomplete units explicitly. "unbalanced" uses observed rows: saturated
        SA regression, BJS imputation on actual untreated observations, or CS
        repeated-cross-section functionals with observed-date covariates and
        scores clustered by original unit (R did's unbalanced convention).
        Balanced input uses balanced comparison conventions; weighted SA/BJS
        use the weighted within-unit systems.
    control_group : {"nevertreated", "notyettreated"}
        CS/ETWFE comparison population. SA/BJS require this option to remain
        nevertreated; reference_group selects the clean reference window separately.
    base_period : {"varying", "universal"}
        CS unaffected pre-periods use t-1 (varying) or g-anticipation-1
        (universal). Affected periods use g-anticipation-1 under either setting.
        Universal CS and SA omit that reference cell. BJS returns effects from
        g-anticipation onward, including any anticipatory effects.
    anticipation : int
        Nonnegative periods of possible effects before actual treatment, default
        0. Clean observations have t < g-anticipation; the clean reference is
        g-anticipation-1. Event times and cohort labels retain actual treatment.
        Cohorts without a clean pre-period raise; future cohorts are never-treated
        only when their anticipation window also starts after the sample.
    weights : str, optional
        Sampling-weight column. Weights must be finite, strictly positive and
        constant within each unit unless explicitly allowed to vary. They enter fitting, aggregation and clustered
        scores; they are not frequency counts or propensity weights. Scaling all
        weights by a common positive constant does not change estimates or SEs.
    allow_time_varying_weights : bool
        Default False preserves strict unit-weight validation. True permits
        observation weights: SA/BJS use row WLS and target-cell weight totals;
        balanced CS uses earlier-comparison-date weights, native unbalanced CS
        observed-date weights. CS cohort shares use first-period unit weights
        for balanced panels and mean observed unit weights for unbalanced panels,
        matching R did. Cluster counts remain physical retained units.
    ess_warn_threshold : float or None
        CS only: warn when any required treated/control stratum has Kish ESS
        below this threshold (default 30). None or 0 disables the warning.
        IPW/DR control weights include propensity odds and trimming. This is
        a support diagnostic, not a coverage guarantee or BJS's treated ESS.
    ess_suppress_threshold : float or None
        CS only: optionally mask low-ESS coefficients and their covariance as
        NaN, preserving the original estimation/aggregation weights. None
        (default) retains every estimate. Post-estimation suppress() provides
        the same operation while preserving the original result for inspection.
    vcov : {"CRV0", "CRV1"}
        Unit clustering. CRV1 applies N/(N-1) only. Normal inference.
    batch_units : int
        Maximum units per numerical batch; no full N-by-effects score store.
    threads : int, optional
        Numerical thread limit; Polars pool is configured before import.
    memory_gb : float, optional
        Total process budget for guards, including current RSS and a reserve.
        Checked before staging and before partition/numerical allocations.
        Source query execution and third-party allocators are not hard capped.
    workdir : path, optional
        Parent of isolated run directories. Weighted or native unbalanced BJS retains
        sorted Parquet until result.cleanup() or result garbage collection,
        to support contrast-specific inference in bounded memory.
    profile : bool
        Validate through disk partitions and return an hdfe-stream Profile,
        without fitting. Profiling requires temporary disk space.
    preprocess_mb : float
        Decoded partition/sort working allowance (decimal MB). Oversized hash
        partitions are recursively split. Source readers/writers have separate
        overhead; this is not a process RSS ceiling.
    avgeffectsby : {"cohort_time", "event_time", "cohort", "all"}
        BJS residual averaging groups. Smart weights are recomputed for every
        requested aggregation. Default corresponds to the authors' cohort-time
        grouping, with unit clusters. Sampling weights also enter smart averaging.
    leaveout : bool
        BJS cluster leave-out adjustment to treated residuals, as in the
        authors' ``leaveout`` option. Singleton contributing groups raise an
        error; broader avgeffectsby groups may make leave-out possible.
    est_method : {"reg", "ipw", "dr"}
        CS outcome regression, normalized IPW, or traditional doubly robust
        estimation. Nuisances are control OLS and unpenalized logit MLE, matching
        R did/DRDID. Balanced comparisons use earlier-date covariates;
        native unbalanced comparisons use each observed row's covariates.
    trim_level : float
        CS IPW/DR upper propensity threshold for controls only, default 0.995.
        Treated units and nuisance-fit samples are retained. Probabilities are
        capped at 1-1e-6; 1 disables control trimming. Ignored by regression.
        Analytic inference holds trim membership fixed, as in R DRDID.
    bootstrap_reps : int
        CS unit multiplier repetitions, default 0 (disabled). At least 2 when
        enabled. Retains repetitions-by-effects draws and cohort-share draws,
        never a full units-by-effects or units-by-repetitions table. Use the
        result's multiplier_confint() for bands; analytic covariance is unchanged.
    bootstrap_seed : int, optional
        Nonnegative RNG seed. Reproducible for a fixed staged unit traversal;
        repartitioning can change assignments while preserving their distribution.
    bootstrap_weights : {"mammen", "rademacher", "webb", "norm"}
        Multiplier sampler reused from hdfe-stream's wild bootstrap. Default
        Rademacher weights match R did/BMisc's multiplier distribution. No enumeration.
    pretrends : int
        BJS clean-placebo lead count, default 0 (disabled). Fits a separate
        clean-sample unit/time WLS regression augmented with the last L clean
        event indicators, actual horizons -anticipation-L..-anticipation-1.
        Retrieve with placebos() or test jointly with pretrend_test().

    specification : {"dynamic", "static"}
        Gardner second-stage specification; see gardner_stream.
    reference_period : int, optional
        Gardner dynamic clean reference event time (default -anticipation-1).

    fixed_effects : {"cohort", "unit"}
        ETWFE intercept specification.
    inference : {"unconditional", "conditional"}
        ETWFE treatment-effect covariance convention.
    max_design_columns : int
        ETWFE expanded saturated-design limit (default 2048).

    reference_group : {"auto", "nevertreated", "lasttreated"}
        Reference policy for all estimators. Auto uses a clean never-treated group if present,
        otherwise the last-treated cohort and dates strictly before g_last-anticipation.
        Last-treated reference effects are not estimated. BJS clean placebos retain
        original adoption dates; Gardner assigns the reference zero second-stage
        design. Explicit nevertreated is strict.
    balance : bool
        CS only: save comparison-specific covariate balance tables at fit time.
    balance_ecdf : bool
        With balance=True, additionally compute sorted empirical-CDF gaps.

    Returns
    -------
    DIDResult, GardnerResult or hdfe_stream.core.profile.Profile
        Cell estimates and joint covariance; ``aggregate()`` gives an event
        study, ``aggregate("simple")`` the post-treatment ATT. Gardner returns
        direct second-stage coefficients without aggregate(). No automatic
        Stata minn suppression is applied.
    """
    if reference_group not in ("auto", "nevertreated", "lasttreated"):
        raise ValueError("reference_group must be auto, nevertreated or lasttreated")
    if not isinstance(balance, bool) or not isinstance(balance_ecdf, bool):
        raise ValueError("balance and balance_ecdf must be boolean")
    if (balance or balance_ecdf) and estimator != "cs":
        raise ValueError("covariate balance diagnostics are available only for CS")
    if balance_ecdf and not balance:
        raise ValueError("balance_ecdf requires balance=True")
    if estimator == "etwfe":
        if (
            base_period != "varying"
            or avgeffectsby != "cohort_time"
            or leaveout
            or est_method != "dr"
            or trim_level != 0.995
            or bootstrap_reps != 0
            or bootstrap_seed is not None
            or bootstrap_weights != "rademacher"
            or pretrends != 0
            or ess_warn_threshold not in (None, 30)
            or ess_suppress_threshold is not None
            or specification != "dynamic"
            or reference_period is not None
        ):
            raise ValueError(
                "ETWFE does not accept CS/BJS/Gardner-specific options or bootstrap options"
            )
        from .etwfe import etwfe_stream

        return etwfe_stream(
            fml,
            data,
            panel=panel,
            cohort=cohort,
            control_group=control_group,
            fixed_effects=fixed_effects,
            inference=inference,
            max_design_columns=max_design_columns,
            vcov=vcov,
            weights=weights,
            allow_time_varying_weights=allow_time_varying_weights,
            panel_mode=panel_mode,
            anticipation=anticipation,
            batch_units=batch_units,
            threads=threads,
            memory_gb=memory_gb,
            preprocess_mb=preprocess_mb,
            workdir=workdir,
            profile=profile,
            reference_group=reference_group,
        )
    if fixed_effects != "cohort" or inference != "unconditional" or max_design_columns != 2048:
        raise ValueError("fixed_effects, inference and max_design_columns are ETWFE options")
    if estimator == "gardner":
        if (
            control_group != "nevertreated"
            or base_period != "varying"
            or avgeffectsby != "cohort_time"
            or leaveout
            or est_method != "dr"
            or trim_level != 0.995
            or bootstrap_reps != 0
            or bootstrap_seed is not None
            or bootstrap_weights != "rademacher"
            or pretrends != 0
            or ess_warn_threshold not in (None, 30)
            or ess_suppress_threshold is not None
        ):
            raise ValueError("Gardner does not accept CS/BJS-specific options or bootstrap options")
        from .gardner import gardner_stream

        return gardner_stream(
            fml,
            data,
            panel=panel,
            cohort=cohort,
            specification=specification,
            reference_period=reference_period,
            vcov=vcov,
            batch_units=batch_units,
            threads=threads,
            memory_gb=memory_gb,
            workdir=workdir,
            profile=profile,
            preprocess_mb=preprocess_mb,
            panel_mode=panel_mode,
            anticipation=anticipation,
            weights=weights,
            allow_time_varying_weights=allow_time_varying_weights,
            reference_group=reference_group,
        )
    if specification != "dynamic" or reference_period is not None:
        raise ValueError("specification and reference_period are Gardner options")
    validate_bootstrap(bootstrap_reps, bootstrap_seed, bootstrap_weights)
    if not isinstance(allow_time_varying_weights, bool):
        raise ValueError("allow_time_varying_weights must be boolean")
    bootstrap_reps = int(bootstrap_reps)
    if estimator != "cs" and bootstrap_reps:
        raise ValueError("unit multiplier bootstrap is currently available only for CS")
    if (
        isinstance(anticipation, (bool, np.bool_))
        or not isinstance(anticipation, Integral)
        or not 0 <= anticipation <= np.iinfo(np.int64).max
    ):
        raise ValueError("anticipation must be a nonnegative integer within the int64 range")
    anticipation = int(anticipation)
    if (
        isinstance(pretrends, (bool, np.bool_))
        or not isinstance(pretrends, Integral)
        or not 0 <= pretrends <= np.iinfo(np.int64).max - anticipation
    ):
        raise ValueError(
            "pretrends must be a nonnegative integer with anticipation+pretrends within int64 range"
        )
    pretrends = int(pretrends)
    if estimator != "bjs" and pretrends:
        raise ValueError("pretrends is available only for BJS")
    if panel_mode not in ("balanced", "complete", "unbalanced"):
        raise ValueError("panel_mode must be 'balanced', 'complete' or 'unbalanced'")
    if estimator not in ("cs", "sunab", "bjs"):
        raise ValueError("estimator must be 'cs', 'sunab', 'bjs', 'gardner' or 'etwfe'")
    if est_method not in ("reg", "ipw", "dr"):
        raise ValueError("est_method must be 'reg', 'ipw' or 'dr'")
    if isinstance(trim_level, bool) or not np.isfinite(trim_level) or not 0 < trim_level <= 1:
        raise ValueError("trim_level must be in (0, 1]")
    if estimator != "cs" and (est_method != "dr" or trim_level != 0.995):
        raise ValueError("est_method and trim_level are CS options")
    if control_group not in ("nevertreated", "notyettreated"):
        raise ValueError("unknown control_group")
    if base_period not in ("varying", "universal"):
        raise ValueError("unknown base_period")
    if estimator != "cs" and control_group != "nevertreated":
        raise ValueError("notyettreated controls are available only for CS")
    if vcov not in ("CRV0", "CRV1"):
        raise ValueError("vcov must be 'CRV0' or 'CRV1'")
    if not isinstance(batch_units, int) or isinstance(batch_units, bool) or batch_units < 1:
        raise ValueError("batch_units must be a positive integer")
    if threads is not None and (
        not isinstance(threads, int) or isinstance(threads, bool) or threads < 1
    ):
        raise ValueError("threads must be a positive integer")
    if isinstance(preprocess_mb, bool) or not np.isfinite(preprocess_mb) or preprocess_mb < 1:
        raise ValueError("preprocess_mb must be finite and at least 1 MB")
    if avgeffectsby not in ("cohort_time", "event_time", "cohort", "all"):
        raise ValueError("unknown avgeffectsby")
    if not isinstance(leaveout, bool):
        raise ValueError("leaveout must be boolean")
    if estimator != "bjs" and (leaveout or avgeffectsby != "cohort_time"):
        raise ValueError("leaveout and avgeffectsby options are available only for BJS")
    ess_warn_threshold = threshold(ess_warn_threshold, "ess_warn_threshold")
    ess_suppress_threshold = threshold(ess_suppress_threshold, "ess_suppress_threshold")
    if estimator != "cs" and (
        ess_suppress_threshold is not None or ess_warn_threshold not in (None, 30)
    ):
        raise ValueError("ESS warning/suppression thresholds are CS options")

    balance_collector = None

    def finish(result):
        if balance_collector is not None:
            result._balance_table = balance_collector.table()
            result.diagnostics["balance_source_passes"] = balance_collector.source_passes
            result.diagnostics["balance_scale"] = (
                "pooled before-weighting SD, held fixed after weighting"
            )
        return finish_result(result, ess_warn_threshold, ess_suppress_threshold)

    budget = MemoryBudget(memory_gb)
    baseline = current_rss_gb() or 0
    with thread_scope(threads):
        lf, yname, design = inspect(
            fml,
            data,
            panel,
            cohort,
            allow_covariates=True,
            weights=weights,
            allow_fe=estimator != "cs",
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
                batch_units,
                budget,
                preprocess_mb,
                design,
                panel_mode,
                anticipation,
                require_never_treated=reference_group == "nevertreated",
                retain_actual_cohort=bool(pretrends),
                allow_time_varying_weights=allow_time_varying_weights,
            )
            from .reference import reference_store

            store, reference_metadata = reference_store(store, reference_group, anticipation)
            if balance:
                from .balance import CSBalanceCollector

                balance_collector = CSBalanceCollector(store, balance_ecdf)
            # Preparation readers/writers are closed. Return their unused Arrow
            # pool pages before reserving inference/JIT buffers (not live data).
            pa.default_memory_pool().release_unused()
            times, gc, N, means = store.times, store.gc, store.n_units, store.means
            groups, cells, bases, controls, sizes = setup(
                estimator, times, gc, control_group, base_period, anticipation
            )
            omitted = []
            if not store.is_balanced:
                supported = []
                for j, (g, t) in enumerate(cells):
                    ti, bi = t - times[0], bases[j]
                    ok = store.cell_counts[g][ti] > 0
                    if estimator == "cs":
                        cc = np.sum(
                            [
                                store.cell_counts[h]
                                for h, take in zip(groups, controls[j], strict=True)
                                if take
                            ],
                            axis=0,
                        )
                        ok = ok and store.cell_counts[g][bi] > 0 and cc[ti] > 0 and cc[bi] > 0
                    if ok:
                        supported.append(j)
                    else:
                        omitted.append(
                            dict(
                                cohort=g,
                                time=t,
                                reason="no observations in a required cohort/date stratum",
                            )
                        )
                cells = [cells[j] for j in supported]
                bases, controls = bases[supported], controls[supported]
                if not cells:
                    raise ValueError("no supported observed-row comparisons")
            sample_diagnostics = dict(
                **reference_metadata,
                balance_requested=balance,
                balance_ecdf=balance_ecdf,
                pretrends=pretrends,
                placebo_cohort_policy="original known adoption dates, including reference/future cohorts"
                if pretrends
                else None,
                bootstrap_reps=bootstrap_reps,
                bootstrap_seed=int(bootstrap_seed) if bootstrap_seed is not None else None,
                bootstrap_weights=bootstrap_weights if bootstrap_reps else None,
                bootstrap_unit_order="staged whole-unit traversal; seed reproducible for fixed partition layout"
                if bootstrap_reps
                else None,
                sampling_weights=weights,
                weight_convention=(
                    "positive time-invariant unit sampling weights"
                    if store.weights_time_invariant
                    else "positive time-varying observation weights"
                )
                if weights is not None
                else "equal unit weights",
                allow_time_varying_weights=allow_time_varying_weights,
                weights_time_invariant=store.weights_time_invariant,
                unit_weight_anchor=store.unit_weight_anchor,
                weight_scale=store.weight_scale / store.weight_normalizer,
                total_unit_weight=float(N * (store.weight_scale / store.weight_normalizer)),
                cohort_weight_totals={
                    str(g): float(v * (store.weight_scale / store.weight_normalizer))
                    for g, v in store.gw.items()
                },
                anticipation=anticipation,
                reference_event_time=-anticipation - 1,
                cohort_sizes={str(g): int(n) for g, n in gc.items()},
                time_min=int(times[0]),
                time_max=int(times[-1]),
                panel_mode=panel_mode,
                balanced_sample=store.is_balanced,
                input_units=store.n_units_input,
                input_rows=store.n_rows,
                dropped_units=store.n_units_input - N,
                dropped_rows=store.n_rows - store.n_rows_used,
                omitted_cells=omitted,
            )
            M, T, G = len(cells), len(times), len(groups)
            rich_regression = estimator != "cs" and (
                design is not None or not store.weights_time_invariant
            )
            B = min(batch_units, N)
            numeric = 8 * (12 * M * M + 4 * B * M + 8 * G * T * T + 8 * T * T + 4 * B * T)
            if pretrends:
                P = T - 1 + pretrends
                numeric += 8 * (8 * P * P + 4 * T * P + 8 * B * P + 8 * B * T)
                numeric += 1024 * len(store.actual_gc) * min(pretrends, T) + 32e6
            if bootstrap_reps:
                numeric += 8 * (
                    6 * bootstrap_reps * M
                    + 2 * max(bootstrap_reps, min(B * bootstrap_reps, MAX_DRAW_ENTRIES))
                )
            if rich_regression:
                K = len(design.names) if design is not None else 0
                P = T - 1 + K + (M if estimator == "sunab" else pretrends)
                numeric += 8 * (8 * P * P + 12 * B * T * (2 * T + K + pretrends))
            elif design is not None:
                K = len(design.names)
                # Nuisance Grams are system-by-K-by-K and row designs are
                # batch-by-time-by-K. Cross-cell scores/covariance above have
                # no covariate axis: neither M-by-M-by-K nor B-by-M-by-K is
                # materialized. Keep generous allowances for four unbalanced
                # outcome strata, Newton information and contrast derivatives.
                numeric += 8 * (8 * M * K * K + 8 * M * T * K + 12 * B * T * K)
            p = Profile(
                model=f"diff-stream {estimator}",
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
                        preprocess_mb * 1e6 + 96e6,
                        preprocess_mb * 1e6 + 256e6,
                    ),
                    Component(
                        "snapshot, hash partitions and sorted leaves",
                        64 * N * T,
                        128 * N * T,
                        where="disk",
                    ),
                ],
                notes={
                    **sample_diagnostics,
                    "n_units": N,
                    "periods": T,
                    "cells": M,
                    "batch_units": B,
                    "validation": "whole-unit disk partitions; no global unit-ID table",
                    "partitions": store.partition_count,
                    "max_partition_depth": store.max_depth,
                    "max_decoded_leaf_bytes": store.max_leaf_bytes,
                    "preprocess_mb": preprocess_mb,
                    "covariate_names": design.names if design is not None else [],
                },
                warnings=[
                    "Heuristic estimate: source query, readers, writers and allocator retention can exceed allowances."
                ],
            )
            if profile:
                return p
            if not budget.fits(numeric / 1e9 + 128e6 / 1e9):
                raise MemoryError(
                    "estimated numerical buffers exceed job headroom; reduce batch_units or request more memory"
                )
            multiplier = (
                StreamingMultiplier(2 * M, bootstrap_reps, bootstrap_seed, bootstrap_weights)
                if bootstrap_reps
                else None
            )
            multiplier_fields = (
                dict(
                    _multiplier_draws=multiplier.draws[:, :M],
                    _multiplier_shares=multiplier.draws[:, M:],
                )
                if multiplier is not None
                else {}
            )
            placebo_fields = {}
            if pretrends:
                placebo = fit_placebos(
                    store, pretrends, anticipation, fml=fml, depvar=yname, panel=panel, vcov=vcov
                )
                placebo.diagnostics.update(sampling_weights=weights, panel_mode=panel_mode)
                placebo_fields = dict(_bjs_placebos=placebo)
            if not store.is_balanced or (store.weighted and estimator != "cs") or rich_regression:
                from .cs_unbalanced import fit_cs_unbalanced
                from .unbalanced import fit_bjs, fit_sunab

                context, cross, share_cov, metadata = None, None, None, {}
                if estimator == "cs":
                    att, counts, V, cross, share_cov, metadata = fit_cs_unbalanced(
                        store,
                        groups,
                        cells,
                        bases,
                        controls,
                        est_method,
                        trim_level,
                        multiplier,
                        balance_collector=balance_collector,
                    )
                elif rich_regression:
                    from . import linear

                    if estimator == "sunab":
                        att, counts, V, model = linear.fit_sunab(store, cells, anticipation)
                    else:
                        att, counts, V, context, model = linear.fit_bjs(
                            store, cells, avgeffectsby, leaveout, workdir, anticipation
                        )
                    metadata.update(
                        regression_backend="streamed weighted unit elimination; bounded joint design",
                        covariate_names=design.names if design is not None else [],
                        regression_columns=[model.names[j] for j in model.keep],
                        absorbed_or_redundant_columns=model.dropped,
                    )
                elif estimator == "sunab":
                    att, counts, V = fit_sunab(store, cells, anticipation)
                else:
                    att, counts, V, context = fit_bjs(
                        store, cells, avgeffectsby, leaveout, workdir, anticipation
                    )
                factor = N / (N - 1) if vcov == "CRV1" else 1.0
                diagnostics = dict(
                    **sample_diagnostics,
                    control_group=control_group,
                    base_period=base_period if estimator == "cs" else f"g-{anticipation + 1}",
                    inference="unit clustered; observed-row analytic scores",
                    periods=T,
                    batch_units=B,
                    preprocess_mb=preprocess_mb,
                    partitions=store.partition_count,
                    max_decoded_leaf_bytes=store.max_leaf_bytes,
                    warnings=[],
                )
                diagnostics.update(metadata)
                if omitted:
                    diagnostics["warnings"].append(
                        "Some cells have no required observed-row stratum; inspect omitted_cells."
                    )
                if context is not None:
                    diagnostics.update(
                        avgeffectsby=avgeffectsby,
                        leaveout=leaveout,
                        aggregation_storage="temporary Parquet; call cleanup() after aggregation",
                    )
                    if np.isnan(V).any():
                        diagnostics["warnings"].append(
                            "Some cell leave-out variances are undefined; aggregate over supported groups before inference."
                        )
                return finish(
                    DIDResult(
                        **placebo_fields,
                        **multiplier_fields,
                        fml=fml,
                        depvar=yname,
                        coefnames=[f"g::{g}:t::{t}" for g, t in cells],
                        beta=att,
                        vcov=V * factor,
                        vcov_type=vcov,
                        df_t=N - 1,
                        all_vcovs={vcov: (V * factor, N - 1)},
                        n_obs=store.n_rows_used,
                        n_clusters={panel[0]: N},
                        estimator=estimator,
                        cells=cells,
                        counts=counts,
                        observed_counts=np.array(
                            [store.cell_counts[g][t - times[0]] for g, t in cells], dtype=float
                        ),
                        cohort_counts=np.array([gc[g] for g, t in cells], dtype=float),
                        n_units=N,
                        diagnostics=diagnostics,
                        share_cross=cross * factor if cross is not None else None,
                        share_cov=share_cov * factor if share_cov is not None else None,
                        bjs_inference=context,
                    )
                )
            if (
                design is not None
                or (store.weighted and estimator == "cs")
                or multiplier is not None
            ):
                att, counts, V, cross, share_cov, metadata = fit_cs(
                    store,
                    groups,
                    cells,
                    bases,
                    controls,
                    est_method,
                    trim_level,
                    multiplier,
                    balance_collector=balance_collector,
                )
                factor = N / (N - 1) if vcov == "CRV1" else 1.0
                return finish(
                    DIDResult(
                        **multiplier_fields,
                        fml=fml,
                        depvar=yname,
                        coefnames=[f"g::{g}:t::{t}" for g, t in cells],
                        beta=att,
                        vcov=V * factor,
                        vcov_type=vcov,
                        df_t=N - 1,
                        all_vcovs={vcov: (V * factor, N - 1)},
                        n_obs=store.n_rows_used,
                        n_clusters={panel[0]: N},
                        estimator=estimator,
                        cells=cells,
                        counts=counts,
                        observed_counts=np.array(
                            [store.cell_counts[g][t - times[0]] for g, t in cells], dtype=float
                        ),
                        cohort_counts=np.array([gc[g] for g, t in cells], dtype=float),
                        n_units=N,
                        share_cross=cross * factor,
                        share_cov=share_cov * factor,
                        diagnostics=dict(
                            **sample_diagnostics,
                            inference="unit clustered; nuisance-adjusted analytic scores",
                            control_group=control_group,
                            base_period=base_period,
                            periods=T,
                            batch_units=B,
                            partitions=store.partition_count,
                            preprocess_mb=preprocess_mb,
                            max_decoded_leaf_bytes=store.max_leaf_bytes,
                            **metadata,
                        ),
                    )
                )
            if estimator == "cs" and est_method != "reg":
                ng = np.array([gc[g] for g, t in cells])
                probability = np.minimum(ng / (ng + controls @ sizes), 1 - 1e-6)
                if np.any(probability >= trim_level):
                    raise ValueError(
                        "no controls remain after propensity trimming in an intercept-only comparison"
                    )
            att, counts, args = fit_means(
                estimator, times, groups, cells, bases, controls, sizes, means, anticipation
            )
            V, cross, share_cov = np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M))
            scatter = np.zeros((G, T, T)) if estimator == "bjs" else None
            for g, y in store.batches():
                idx = np.searchsorted(groups, g)
                if estimator == "bjs":
                    for k in np.unique(idx):
                        centered = y[idx == k] - means[k]
                        use = (
                            times < groups[k] - anticipation
                            if groups[k]
                            else np.ones(T, dtype=bool)
                        )
                        centered -= centered[:, use].mean(axis=1, keepdims=True)
                        scatter[k] += centered.T @ centered
                else:
                    scores, share = change_scores(idx, y, *args, sizes)
                    if estimator == "cs":
                        cross += scores.T @ share
                        share_cov += share.T @ share
                    V += scores.T @ scores
            bjs_inference = None
            if estimator == "bjs":
                _, _, Z, theta, H = args
                bjs_inference = BJSInference(
                    np.array(groups),
                    sizes,
                    times,
                    cells,
                    means,
                    Z,
                    theta,
                    H,
                    scatter,
                    avgeffectsby,
                    leaveout,
                    anticipation,
                )
                V = bjs_inference.covariance(
                    np.eye(M), allow_unidentified=avgeffectsby != "cohort_time"
                )
            factor = N / (N - 1) if vcov == "CRV1" else 1.0
            diagnostics = {
                **sample_diagnostics,
                "control_group": control_group,
                "base_period": base_period if estimator == "cs" else f"g-{anticipation + 1}",
                "est_method": est_method if estimator == "cs" else None,
                "inference": "unit clustered; normal; "
                + ("BJS smart residual averaging" if estimator == "bjs" else "analytic scores"),
                "avgeffectsby": avgeffectsby if estimator == "bjs" else None,
                "leaveout": leaveout if estimator == "bjs" else None,
                "periods": T,
                "batch_units": B,
                "preprocess_mb": preprocess_mb,
                "partitions": store.partition_count,
                "max_decoded_leaf_bytes": store.max_leaf_bytes,
                "warnings": (
                    ["Small BJS cohorts: consider leaveout=True and broader avgeffectsby groups."]
                    if estimator == "bjs" and not leaveout and min(gc[g] for g in gc if g) < 20
                    else []
                ),
            }
            if estimator == "cs":
                diagnostics["comparisons"] = [
                    dict(
                        cohort=g,
                        time=t,
                        cells=[j],
                        effective_treated=float(gc[g]),
                        effective_controls=float(controls[j] @ sizes),
                    )
                    for j, (g, t) in enumerate(cells)
                ]
            if np.isnan(V).any():
                diagnostics["warnings"].append(
                    "Some cell leave-out variances are undefined; aggregate over supported groups before inference."
                )
            return finish(
                DIDResult(
                    **placebo_fields,
                    fml=fml,
                    depvar=yname,
                    coefnames=[f"g::{g}:t::{t}" for g, t in cells],
                    beta=att,
                    vcov=V * factor,
                    vcov_type=vcov,
                    df_t=N - 1,
                    all_vcovs={vcov: (V * factor, N - 1)},
                    n_obs=store.n_rows_used,
                    n_clusters={panel[0]: N},
                    estimator=estimator,
                    cells=cells,
                    counts=counts,
                    observed_counts=np.array(
                        [store.cell_counts[g][t - times[0]] for g, t in cells], dtype=float
                    ),
                    cohort_counts=np.array([gc[g] for g, t in cells], dtype=float),
                    n_units=N,
                    diagnostics=diagnostics,
                    share_cross=cross * factor if estimator == "cs" else None,
                    share_cov=(share_cov - np.outer(counts, counts) / N) * factor
                    if estimator == "cs"
                    else None,
                    bjs_inference=bjs_inference,
                )
            )


def bjs_stream(fml, data, **kwargs):
    """Borusyak–Jaravel–Spiess imputation; see :func:`diff_stream`."""
    return diff_stream(fml, data, estimator="bjs", **kwargs)


def sunab_stream(fml, data, **kwargs):
    """Sun–Abraham saturated interactions; see :func:`diff_stream`."""
    return diff_stream(fml, data, estimator="sunab", **kwargs)


def cs_stream(fml, data, **kwargs):
    """Unconditional Callaway–Sant'Anna; see :func:`diff_stream`."""
    return diff_stream(fml, data, estimator="cs", **kwargs)

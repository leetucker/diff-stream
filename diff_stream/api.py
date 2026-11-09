"""The panel estimators: Callaway-Sant'Anna, Sun-Abraham and BJS imputation."""

from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import pyarrow as pa
from hdfe_stream.core.profile import Component, Profile
from stream_limits import MemoryBudget, current_rss_gb, detect_cpus, thread_scope

from . import _docs, _options
from ._core.multiplier import MAX_DRAW_ENTRIES, StreamingMultiplier, validate_bootstrap
from ._core.partition import preprocess_budget_mb, staging_bytes
from ._present import present, public_profile
from .bjs import BJSInference
from .bjs_placebo import fit_placebos
from .cohort_scores import CohortScoreCovariance
from .cs import fit_cs
from .effective_sample import finish_result, threshold
from .estimators import change_scores, fit_means, setup
from .results import DIDResult
from .store import PanelStore, inspect


def _fit(
    fml,
    data,
    *,
    panel,
    cohort=None,
    treatment=None,
    estimator="cs",
    control_group="nevertreated",
    base_period="varying",
    vcov="CRV0",
    batch_units=None,
    threads=None,
    memory_gb=None,
    memory_reserve=None,
    workdir=None,
    profile=False,
    preprocess_mb=None,
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
    fe_solver="auto",
):
    """Fit any panel estimator from internal option names.

    The public functions (did_callaway_santanna, ..., did_extended_twfe) and
    refit_bootstrap translate their options with ``_options.internal`` and
    call this. The two-stage and extended-TWFE estimators are dispatched to
    their own modules.
    """
    if reference_group not in ("auto", "nevertreated", "lasttreated"):
        raise ValueError("reference_group must be auto, nevertreated or lasttreated")
    if not isinstance(balance, bool) or not isinstance(balance_ecdf, bool):
        raise ValueError("covariate_balance and balance_distributions must be True or False")
    if (balance or balance_ecdf) and estimator != "cs":
        raise ValueError("covariate balance diagnostics are available only for CS")
    if balance_ecdf and not balance:
        raise ValueError("balance_distributions requires covariate_balance=True")
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
        from .etwfe import _extended_twfe

        return _extended_twfe(
            fml,
            data,
            panel=panel,
            cohort=cohort,
            treatment=treatment,
            control_group=control_group,
            fixed_effects=fixed_effects,
            inference=inference,
            max_design_columns=max_design_columns,
            vcov=vcov,
            weights=weights,
            allow_time_varying_weights=allow_time_varying_weights,
            panel_mode=panel_mode,
            anticipation=anticipation,
            batch_units=batch_units or 4096,
            threads=threads,
            memory_gb=memory_gb,
            memory_reserve=memory_reserve,
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
        from .gardner import _two_stage

        return _two_stage(
            fml,
            data,
            panel=panel,
            cohort=cohort,
            treatment=treatment,
            specification=specification,
            reference_period=reference_period,
            vcov=vcov,
            batch_units=batch_units or 4096,
            threads=threads,
            memory_gb=memory_gb,
            memory_reserve=memory_reserve,
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
        raise ValueError("allow_time_varying_weights must be True or False")
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
            "pretrend_leads must be a nonnegative integer with anticipation+pretrend_leads within int64 range"
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
        raise ValueError("propensity_trim must be in (0, 1]")
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
    if batch_units is not None and (
        not isinstance(batch_units, int) or isinstance(batch_units, bool) or batch_units < 1
    ):
        raise ValueError("batch_units must be a positive integer or None")
    if threads is not None and (
        not isinstance(threads, int) or isinstance(threads, bool) or threads < 1
    ):
        raise ValueError("n_threads must be a positive integer")
    if avgeffectsby not in ("cohort_time", "event_time", "cohort", "all"):
        raise ValueError(
            "residual_averaging must be 'cohort_time', 'event_time', 'cohort' or 'all'"
        )
    if not isinstance(leaveout, bool):
        raise ValueError("leave_out must be True or False")
    if estimator != "bjs" and (leaveout or avgeffectsby != "cohort_time"):
        raise ValueError("leaveout and avgeffectsby options are available only for BJS")
    ess_warn_threshold = threshold(ess_warn_threshold, "effective_sample_warning")
    ess_suppress_threshold = threshold(ess_suppress_threshold, "min_effective_sample")
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
        result = finish_result(result, ess_warn_threshold, ess_suppress_threshold)
        return present(result, store.period_labels)

    budget = MemoryBudget(memory_gb, memory_reserve)
    automatic_preprocess = preprocess_mb is None
    preprocess_mb = preprocess_budget_mb(preprocess_mb, budget)
    baseline = current_rss_gb() or 0
    with thread_scope(threads):
        if fe_solver not in ("auto", "dense", "iterative"):
            raise ValueError("fixed_effect_solver must be 'auto', 'dense' or 'iterative'")
        lf, yname, design, fe_terms, iterative = inspect(
            fml,
            data,
            panel,
            cohort,
            treatment=treatment,
            allow_covariates=True,
            weights=weights,
            allow_fe=estimator != "cs",
            fe_solver=fe_solver,
            keep_fe_columns=estimator == "bjs",
        )
        absorbed = fe_terms if iterative else None
        if automatic_preprocess:
            # counting the levels of large fixed effects used memory: size the
            # default from what is left now
            preprocess_mb = preprocess_budget_mb(None, budget)
        if not budget.fits((preprocess_mb * 1e6 + 96e6) / 1e9):
            raise MemoryError(
                "preprocessing buffers exceed job headroom before staging; reduce preprocess_mb or request more memory"
            )
        if workdir is not None:
            Path(workdir).mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="diff-stream-", dir=workdir) as run:
            try:
                store = PanelStore(
                    lf,
                    run,
                    batch_units or 16384,
                    budget,
                    preprocess_mb,
                    design,
                    panel_mode,
                    anticipation,
                    require_never_treated=reference_group == "nevertreated",
                    retain_actual_cohort=bool(pretrends),
                    allow_time_varying_weights=allow_time_varying_weights,
                )
            except ValueError as error:
                if "256" in str(error) and "|" in fml and absorbed is None:
                    raise ValueError(
                        f"{error}; fixed_effect_solver='iterative' absorbs the additional "
                        "fixed effects instead"
                    ) from None
                raise
            from .reference import reference_store

            store, reference_metadata = reference_store(store, reference_group, anticipation)
            unimputed = {}
            if estimator == "bjs" and fe_terms:
                from .absorbed import drop_unimputable

                store, unimputed = drop_unimputable(store, fe_terms, anticipation)
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
                **unimputed,
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
                design is not None
                or not store.weights_time_invariant
                or absorbed is not None
                or (estimator == "bjs" and not store.is_balanced)
            )

            # Numerical buffers grow with the batch; an automatic batch starts
            # large (fast) and halves until it fits the memory left.
            def buffers(B):
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
                return numeric

            B = min(batch_units or 16384, N)
            if batch_units is None:
                while B > 256 and not budget.fits(buffers(B) / 1e9 + 128e6 / 1e9):
                    B //= 2
            numeric = buffers(B)
            store.batch_units = B
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
                        staging_bytes(preprocess_mb, store) + 96e6,
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
                if absorbed is not None:
                    from .absorbed import fit_placebos_absorbed

                    placebo = fit_placebos_absorbed(
                        store,
                        pretrends,
                        anticipation,
                        absorbed,
                        fml=fml,
                        depvar=yname,
                        panel=panel,
                        vcov=vcov,
                        threads=threads,
                    )
                else:
                    placebo = fit_placebos(
                        store,
                        pretrends,
                        anticipation,
                        fml=fml,
                        depvar=yname,
                        panel=panel,
                        vcov=vcov,
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

                    if absorbed is not None and estimator == "sunab":
                        from .absorbed import fit_sunab_absorbed

                        att, counts, V, info = fit_sunab_absorbed(
                            store, cells, absorbed, threads=threads
                        )
                        model = None
                        metadata.update(info)
                    elif estimator == "sunab":
                        att, counts, V, model = linear.fit_sunab(store, cells, anticipation)
                    else:
                        if absorbed is not None:
                            from .absorbed import fit_bjs_absorbed

                            att, counts, V, context, info = fit_bjs_absorbed(
                                store,
                                cells,
                                absorbed,
                                avgeffectsby,
                                leaveout,
                                workdir,
                                anticipation,
                                threads=threads,
                            )
                            model = None
                            metadata.update(info)
                        else:
                            att, counts, V, context, model = linear.fit_bjs(
                                store, cells, avgeffectsby, leaveout, workdir, anticipation
                            )
                    if model is not None:
                        metadata.update(
                            regression_backend="streamed weighted unit elimination; bounded joint design",
                            regression_columns=[model.names[j] for j in model.keep],
                            absorbed_or_redundant_columns=model.dropped,
                        )
                    metadata.update(
                        covariate_names=design.names if design is not None else [],
                        fixed_effect_solver="iterative" if absorbed is not None else "dense",
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
                        _share_cross=cross * factor if cross is not None else None,
                        _share_cov=share_cov * factor if share_cov is not None else None,
                        _bjs_inference=context,
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
                        _share_cross=cross * factor,
                        _share_cov=share_cov * factor,
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
            # Cell scores are zero outside a unit's cohort's cells; accumulate
            # by cohort. Shares are uncentered here and centered below.
            by_cohort = CohortScoreCovariance(args[0]) if estimator != "bjs" else None
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
                    scores, _ = change_scores(idx, y, *args, sizes)
                    by_cohort.add(idx, scores)
            if by_cohort is not None:
                V, cross, share_cov = by_cohort.blocks()
                if estimator != "cs":
                    cross, share_cov = np.zeros((M, M)), np.zeros((M, M))
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
                    _share_cross=cross * factor if estimator == "cs" else None,
                    _share_cov=(share_cov - np.outer(counts, counts) / N) * factor
                    if estimator == "cs"
                    else None,
                    _bjs_inference=bjs_inference,
                )
            )


def _public(estimator, options):
    """Call _fit with a public function's arguments (its locals())."""
    options = dict(options)
    fml, data = options.pop("fml"), options.pop("data")
    return public_profile(_fit(fml, data, estimator=estimator, **_options.internal(options)))


@_docs.document
def did_callaway_santanna(
    fml,
    data,
    *,
    panel=None,
    time=None,
    cohort=None,
    treatment=None,
    control_group="never_treated",
    base_period="varying",
    adjustment="doubly_robust",
    propensity_trim=0.995,
    anticipation=0,
    weights=None,
    allow_time_varying_weights=False,
    incomplete_units="error",
    last_treated_controls="if_needed",
    vcov="CRV0",
    multiplier_bootstrap_reps=0,
    multiplier_bootstrap_seed=None,
    multiplier_bootstrap_weights="rademacher",
    effective_sample_warning=30,
    min_effective_sample=None,
    covariate_balance=False,
    balance_distributions=False,
    n_threads=None,
    max_memory_gb=None,
    memory_reserve=None,
    workdir=None,
    profile=False,
    batch_units=None,
    preprocess_mb=None,
):
    """Callaway and Sant'Anna's group-time average treatment effects.

    Estimates the average effect for each cohort (units first treated in the
    same period) in each period, by comparing its outcome changes with those
    of a control group, then aggregates them with ``aggregate()``. Covariates
    enter by outcome regression, inverse probability weighting or the doubly
    robust combination of Sant'Anna and Zhao (2020), as in R's did.

    Callaway, B. and Sant'Anna, P. H. C. (2021). Difference-in-differences
    with multiple time periods. *Journal of Econometrics* 225(2), 200-230.

    Sant'Anna, P. H. C. and Zhao, J. (2020). Doubly robust
    difference-in-differences estimators. *Journal of Econometrics* 219(1),
    101-122.

    Parameters
    ----------
    fml : str
        ``"y ~ 1"``, or ``"y ~ x + C(region)"`` with covariates (up to 256
        design columns), measured in the comparison's base period.
    {DATA}
    control_group : {"never_treated", "not_yet_treated"}, default "never_treated"
        The comparison units: never-treated units only, or also units not yet
        treated by the period compared.
    base_period : {"varying", "universal"}, default "varying"
        The pre-treatment comparison period: the period just before each
        pre-treatment cell ("varying"), or the last clean period before
        treatment for every cell ("universal"). Post-treatment cells use the
        latter either way.
    adjustment : {"doubly_robust", "regression", "ipw"}, default "doubly_robust"
        How covariates enter: outcome regression on the controls, inverse
        probability weighting with a logit propensity score, or both.
    propensity_trim : float, default 0.995
        Controls with a propensity score above this are dropped from IPW and
        doubly robust comparisons (treated units are kept). 1 disables it.
    {ANTICIPATION}
    {WEIGHTS}
    {INCOMPLETE_UNITS}
        With "keep", comparisons use each period's observed rows, as R's did
        does for unbalanced panels.
    {LAST_TREATED}
    {VCOV}
    multiplier_bootstrap_reps : int, default 0
        Draws of the multiplier bootstrap, which perturbs each unit's
        influence function instead of refitting (R did's default inference).
        It gives simultaneous confidence bands through
        ``multiplier_confint()``; the analytic standard errors are unchanged.
        0 skips it. For a bootstrap that refits the model, see
        ``refit_bootstrap``.
    multiplier_bootstrap_seed : int, optional
        Seed for the multiplier draws.
    multiplier_bootstrap_weights : {"rademacher", "mammen", "webb", "norm"}, default "rademacher"
        The distribution of the multiplier draws.
    effective_sample_warning : float or None, default 30
        Warn (``LowEffectiveSampleWarning``) when a comparison's treated or
        control group has an effective sample size (Kish) below this. None
        turns the warning off.
    min_effective_sample : float, optional
        Report coefficients whose effective sample is below this as NaN.
        ``suppress()`` does the same after fitting.
    covariate_balance : bool, default False
        Save covariate balance tables for each comparison, before and after
        propensity weighting, for ``covariate_balance()``.
    balance_distributions : bool, default False
        With ``covariate_balance``, also compare the covariates' empirical
        distributions.
    {RESOURCES}

    Returns
    -------
    DIDResult or Profile
        ``cohort_time()`` lists the group-time effects; ``aggregate()`` gives
        an event study, ``aggregate("overall")`` the overall ATT.
    """
    options = dict(locals())
    if time is not None:
        options = _cross_sections(options)
    elif panel is None:
        raise ValueError("give panel=(unit, time), or time= for repeated cross sections")
    options.pop("time")
    return _cross_section_labels(_public("cs", options), time)


def _cross_sections(options):
    """Repeated cross sections: every row is its own unit, as in R did (panel=FALSE)."""
    if options["panel"] is not None:
        raise ValueError(
            "give panel=(unit, time) for a panel or time= for cross sections, not both"
        )
    if options["incomplete_units"] != "error":
        raise ValueError("incomplete_units does not apply to repeated cross sections")
    from .store import source

    frame = source(options["data"])
    name = "observation"
    while name in frame.collect_schema().names():
        name = "_" + name
    options.update(
        data=frame.with_row_index(name),
        panel=(name, options["time"]),
        incomplete_units="keep",
    )
    return options


def _cross_section_labels(result, time):
    if isinstance(result, DIDResult):
        result.diagnostics["repeated_cross_sections"] = time is not None
        if time is not None:
            result.diagnostics["incomplete_units"] = None
            result.n_clusters = {"observation": result.n_units}
    return result


@_docs.document
def did_sun_abraham(
    fml,
    data,
    *,
    panel,
    cohort=None,
    treatment=None,
    anticipation=0,
    weights=None,
    allow_time_varying_weights=False,
    incomplete_units="error",
    last_treated_controls="if_needed",
    fixed_effect_solver="auto",
    vcov="CRV0",
    n_threads=None,
    max_memory_gb=None,
    memory_reserve=None,
    workdir=None,
    profile=False,
    batch_units=None,
    preprocess_mb=None,
):
    """Sun and Abraham's interaction-weighted event study.

    Regresses the outcome on a full set of cohort-by-period treatment
    indicators, with unit and time fixed effects, so that each cohort's
    effects are estimated separately; ``aggregate()`` then averages them by
    event time with cohort-share weights. The period before treatment
    (before ``cohort - anticipation``) is the omitted reference.

    Sun, L. and Abraham, S. (2021). Estimating dynamic treatment effects in
    event studies with heterogeneous treatment effects. *Journal of
    Econometrics* 225(2), 175-199.

    Parameters
    ----------
    fml : str
        ``"y ~ 1"``; covariates ``"y ~ x"``; additional fixed effects after
        a bar, ``"y ~ x | region^time"``. Unit and time fixed effects are
        always included.
    {DATA}
    {ANTICIPATION}
    {WEIGHTS}
    {INCOMPLETE_UNITS}
    {LAST_TREATED}
    {FIXED_EFFECT_SOLVER}
    {VCOV}
    {RESOURCES}

    Returns
    -------
    DIDResult or Profile
        ``cohort_time()`` lists the cohort-period coefficients;
        ``aggregate()`` gives the event study.
    """
    return _public("sunab", locals())


@_docs.document
def did_bjs(
    fml,
    data,
    *,
    panel,
    cohort=None,
    treatment=None,
    residual_averaging="cohort_time",
    leave_out=False,
    pretrend_leads=0,
    anticipation=0,
    weights=None,
    allow_time_varying_weights=False,
    incomplete_units="error",
    last_treated_controls="if_needed",
    fixed_effect_solver="auto",
    vcov="CRV0",
    n_threads=None,
    max_memory_gb=None,
    memory_reserve=None,
    workdir=None,
    profile=False,
    batch_units=None,
    preprocess_mb=None,
):
    """Borusyak, Jaravel and Spiess's imputation estimator.

    Fits unit and time fixed effects (and any covariates) on the untreated
    observations only, imputes each treated observation's untreated outcome,
    and averages the differences. The standard errors are the authors'
    conservative ones, as in their Stata command ``did_imputation``.

    Borusyak, K., Jaravel, X. and Spiess, J. (2024). Revisiting event-study
    designs: robust and efficient estimation. *Review of Economic Studies*
    91(6), 3253-3285.

    Parameters
    ----------
    fml : str
        ``"y ~ 1"``; covariates ``"y ~ x"``; additional fixed effects after
        a bar, ``"y ~ x | region^time"``. Unit and time fixed effects are
        always included. Treated observations whose fixed-effect level has no
        untreated observations cannot be imputed: they are dropped, with an
        ``UnimputedObservationsWarning``, and ``fit.diagnostics`` reports
        them (``unimputed_observations``, ``unimputed_cells``).
    {DATA}
    residual_averaging : {"cohort_time", "event_time", "cohort", "all"}, default "cohort_time"
        The groups within which the standard errors average treated
        residuals (the authors' ``avgeffectsby``). Coarser groups give more
        conservative standard errors and need fewer observations per group.
    leave_out : bool, default False
        Leave each unit out of its own group's average residual (the authors'
        ``leaveout``), which reduces the downward bias in small groups.
    pretrend_leads : int, default 0
        For a pre-trend test: fit a separate regression on the untreated
        observations with indicators for the last ``pretrend_leads`` periods
        before treatment. ``placebos()`` returns those coefficients and
        ``pretrend_test()`` tests that they are jointly zero. The treatment
        effects are unchanged. 0 skips it.
    {ANTICIPATION}
    {WEIGHTS}
    {INCOMPLETE_UNITS}
    {LAST_TREATED}
    {FIXED_EFFECT_SOLVER}
    {VCOV}
    {RESOURCES}

    Returns
    -------
    DIDResult or Profile
        ``cohort_time()`` lists the imputed effects by cohort and period;
        ``aggregate()`` gives the event study, ``aggregate("overall")`` the
        overall ATT. Weighted, unbalanced or covariate fits, and fits with
        additional fixed effects, keep temporary files for aggregation until
        ``cleanup()``.
    """
    return _public("bjs", locals())

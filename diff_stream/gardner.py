"""Gardner's two-stage DiD with bounded unit elimination and adjusted inference."""

from copy import copy
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import pyarrow as pa
from hdfe_stream.core.influence import ClusterScoreAccumulator
from hdfe_stream.core.profile import Component, Profile
from hdfe_stream.core.results_base import RegressionResult
from stream_limits import MemoryBudget, current_rss_gb, detect_cpus, thread_scope

from . import _docs, _options
from ._core import twostage_kernels
from ._core.nuisance import solve_checked
from ._core.partition import preprocess_budget_mb, staging_bytes
from ._core.within import row_sums
from ._present import internal, label_frame, labels_of, present, public_profile
from .cs import RETAIN_SHARE, RetainedBatches
from .linear import RowDesign
from .store import PanelStore, inspect


@dataclass(kw_only=True)
class TwoStageResult(RegressionResult):
    """Direct second-stage coefficients, rather than cohort-time estimates.

    Selection returns fitted coefficients without changing either regression.
    No temporary store survives estimation.
    """

    n_units: int
    _support: pl.DataFrame = field(repr=False)
    _internal: dict | None = field(default=None, repr=False)
    _periods: object = field(default=None, repr=False)
    estimator = "gardner"
    cells = ()
    _normal_inference = True

    def support(self):
        """Observed cohort/date counts and original-scale weights per coefficient."""
        return label_frame(self._support, labels_of(self))

    def contributions(self):
        """Support table used by the common pre-treatment Wald helper."""
        return self.support()

    def select(self, horizons):
        """Select fitted event-time coefficients, keeping their joint covariance."""
        if internal(self)["specification"] != "dynamic":
            raise ValueError("horizon selection requires an event-study fit (event_study=True)")
        try:
            requested = list(horizons)
        except TypeError as exc:
            raise ValueError("horizons must contain fitted integer event times") from exc
        if not requested or any(
            isinstance(h, bool) or not isinstance(h, Integral) for h in requested
        ):
            raise ValueError("horizons must contain fitted integer event times")
        names = [f"event::{int(h)}" for h in sorted(set(requested))]
        if not set(names) <= set(self.coefnames):
            raise ValueError(
                "requested horizons were not estimated (possibly the reference period)"
            )
        ix = [self.coefnames.index(name) for name in names]
        out = copy(self)
        out.coefnames = names
        out.beta = self.beta[ix].copy()
        out.vcov = self.vcov[np.ix_(ix, ix)].copy()
        out.all_vcovs = {self.vcov_type: (out.vcov, self.df_t)}
        out._support = self._support.filter(pl.col("aggregate").is_in(names))
        out.diagnostics = {
            **self.diagnostics,
            "selected_horizons": sorted(set(map(int, requested))),
        }
        return out

    def pretrend_test(self, *, level="auto", horizons=None):
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


def second_stage_support(store, specification, reference, anticipation):
    """The second stage's coefficients: (support table rows, event times, names).

    Static fits have one indicator for affected observations (t >=
    g-anticipation); dynamic fits one per observed event time except the
    reference.
    """
    support = []
    scale = store.weight_scale / store.weight_normalizer
    for g, counts in sorted(store.cell_counts.items()):
        if not g:
            continue
        for ti, t in enumerate(store.times):
            event = int(t) - g
            if not counts[ti] or (specification == "static" and event < -anticipation):
                continue
            if specification == "dynamic" and event == reference:
                continue
            support.append(
                dict(
                    aggregate="ATT" if specification == "static" else f"event::{event}",
                    cohort=int(g),
                    time=int(t),
                    event_time=event,
                    observed_units=int(counts[ti]),
                    weight_total=float(store.cell_weights[g][ti] * scale),
                )
            )
    if not support:
        raise ValueError("no supported Gardner second-stage coefficients")
    horizons = sorted({r["event_time"] for r in support}) if specification == "dynamic" else []
    names = [f"event::{h}" for h in horizons] if specification == "dynamic" else ["ATT"]
    return support, horizons, names


def _fit(store, specification, reference, anticipation):
    model = RowDesign(store)
    compiled = twostage_kernels.ENABLED
    source = RetainedBatches(
        model.raw_batches,
        retain=compiled
        and store.n_units * store.periods * 8 * (2 + len(model.names))
        <= RETAIN_SHARE * store.budget.headroom_gb() * 1e9,
    )
    Q, q = model.reduce(
        clean=lambda g: (g[:, None] == 0) | (store.times < g[:, None] - anticipation),
        source=source,
    )
    support, horizons, names = second_stage_support(store, specification, reference, anticipation)
    P, M = len(model.keep), len(names)
    theta = solve_checked(Q, q, "Gardner untreated covariate/fixed-effect design is unidentified")

    def batches():
        for g, y, X, w in model.batches():
            obs, _, xm, u1, score1 = model.residuals(
                g, y, X, w, theta, anticipation, label="Gardner"
            )
            event = store.times[None, :] - g[:, None]
            if specification == "static":
                Z = ((g[:, None] != 0) & (event >= -anticipation))[:, :, None]
            else:
                Z = (g[:, None, None] != 0) & (event[:, :, None] == np.asarray(horizons))
            Z = (Z & obs[:, :, None]).astype(np.float64)
            yield X - xm[:, None], Z, w, u1, score1

    if compiled:
        kernels = twostage_kernels.TwoStageKernels(store.times, anticipation)
        dynamic = specification == "dynamic"
        marks = np.asarray(horizons if dynamic else [0], dtype=np.int64)
        mass, rhs, C = kernels.second_stage(source, model.keep, theta, marks, dynamic)
        if np.any(mass <= 0):
            raise ValueError("Gardner second-stage design is unidentified")
        beta = rhs / mass
        H = solve_checked(Q, C.T, "Gardner first-stage inference design is unidentified")
        accumulator = ClusterScoreAccumulator()
        for raw in source():
            accumulator.add_clusters(
                kernels.scores(raw, model.keep, theta, marks, dynamic, beta, H, mass)
            )
        return beta, accumulator.sums, names, pl.DataFrame(support), model

    # Unit first-stage scores vanish exactly after clean-sample elimination.
    # The Schur-complement cross derivative uses X - clean-unit-mean(X),
    # even though the second-stage regression itself does not demean Z.
    mass, rhs, C = np.zeros(M), np.zeros(M), np.zeros((M, P))
    for xc, Z, w, u1, _ in batches():
        flat = Z.reshape(-1, M)
        mass += flat.T @ w.ravel()
        rhs += flat.T @ (w * u1).ravel()
        C += flat.T @ (w.reshape(-1, 1) * xc.reshape(-1, xc.shape[2]))
    if np.any(mass <= 0):
        raise ValueError("Gardner second-stage design is unidentified")
    beta = rhs / mass  # Supported indicators are mutually exclusive; no intercept.
    H = solve_checked(Q, C.T, "Gardner first-stage inference design is unidentified")
    accumulator = ClusterScoreAccumulator()
    for _, Z, w, u1, score1 in batches():
        u2 = u1 - Z @ beta
        score2 = row_sums(w * u2, Z)
        accumulator.add_clusters((score2 - score1 @ H) / mass)
    return beta, accumulator.sums, names, pl.DataFrame(support), model


def _two_stage(
    fml,
    data,
    *,
    panel,
    cohort=None,
    treatment=None,
    specification="dynamic",
    reference_period=None,
    vcov="CRV0",
    batch_units=4096,
    threads=None,
    memory_gb=None,
    memory_reserve=None,
    workdir=None,
    profile=False,
    preprocess_mb=None,
    panel_mode="balanced",
    anticipation=0,
    weights=None,
    allow_time_varying_weights=False,
    reference_group="auto",
    fe_solver="auto",
):
    """Two-stage DiD from internal option names; see did_two_stage.

    Static fits estimate one indicator for affected observations (t >=
    g-anticipation); dynamic fits every observed event time except the
    reference. There is no second-stage intercept: the omitted reference,
    never-treated rows and a last-treated reference cohort (with its leads)
    have a zero second-stage design. The covariance clusters unit scores
    corrected for the estimated first stage.
    """
    if specification not in ("static", "dynamic"):
        raise ValueError("specification must be 'static' or 'dynamic'")
    if reference_group not in ("auto", "nevertreated", "lasttreated"):
        raise ValueError("reference_group must be auto, nevertreated or lasttreated")
    if (
        isinstance(anticipation, (bool, np.bool_))
        or not isinstance(anticipation, Integral)
        or not 0 <= anticipation < np.iinfo(np.int64).max
    ):
        raise ValueError("anticipation must be a nonnegative integer within the int64 range")
    anticipation = int(anticipation)
    if reference_period is not None and (
        specification != "dynamic"
        or isinstance(reference_period, (bool, np.bool_))
        or not isinstance(reference_period, Integral)
        or not -np.iinfo(np.int64).max <= reference_period < -anticipation
    ):
        raise ValueError(
            "reference_period must be a clean integer event time, with event_study=True"
        )
    reference = -anticipation - 1 if reference_period is None else int(reference_period)
    if vcov not in ("CRV0", "CRV1"):
        raise ValueError("vcov must be 'CRV0' or 'CRV1'")
    if panel_mode not in ("balanced", "complete", "unbalanced"):
        raise ValueError("panel_mode must be 'balanced', 'complete' or 'unbalanced'")
    for value, name in [(batch_units, "batch_units"), (threads, "n_threads")]:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    if batch_units is None:
        raise ValueError("batch_units must be a positive integer")
    if not isinstance(allow_time_varying_weights, bool):
        raise ValueError("allow_time_varying_weights must be boolean")
    budget = MemoryBudget(memory_gb, memory_reserve)
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
            allow_fe=True,
            fe_solver=fe_solver,
        )
        if not budget.fits((preprocess_mb * 1e6 + 96e6) / 1e9):
            raise MemoryError(
                "preprocessing buffers exceed job headroom before staging; reduce preprocess_mb or request more memory"
            )
        if workdir is not None:
            Path(workdir).mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="diff-stream-gardner-", dir=workdir) as run:
            store = PanelStore(
                lf,
                run,
                batch_units,
                budget,
                preprocess_mb,
                design,
                panel_mode,
                anticipation,
                allow_time_varying_weights=allow_time_varying_weights,
                require_never_treated=reference_group == "nevertreated",
            )
            from .reference import reference_store

            store, reference_metadata = reference_store(store, reference_group, anticipation)
            pa.default_memory_pool().release_unused()
            T, B = store.periods, min(batch_units, store.n_units)
            K = T - 1 + (len(design.names) if design is not None else 0)
            events = {
                int(t) - g
                for g, counts in store.cell_counts.items()
                if g
                for t, n in zip(store.times, counts, strict=True)
                if n
            }
            if specification == "dynamic" and reference not in events:
                raise ValueError("reference_period is not observed in the retained treated cohorts")
            M = len(events - {reference}) if specification == "dynamic" else 1
            numeric = 8 * (10 * K * K + 6 * M * M + 8 * K * M + 12 * B * T * (K + M + T))
            diagnostics = dict(
                **reference_metadata,
                specification=specification,
                aggregation="event_time" if specification == "dynamic" else "static",
                anticipation=anticipation,
                reference_event_time=reference if specification == "dynamic" else None,
                reference_second_stage="zero for designated reference cohorts, including their leads",
                time_min=int(store.times[0]),
                time_max=int(store.times[-1]),
                base_period="zero second-stage design for omitted reference and never-treated observations; no intercept",
                panel_mode=panel_mode,
                balanced_sample=store.is_balanced,
                n_units_input=store.n_units_input,
                dropped_units=store.n_units_input - store.n_units,
                n_rows_used=store.n_rows_used,
                sampling_weights=weights,
                weights_time_invariant=store.weights_time_invariant,
                weight_convention="observation WLS in both stages; unit clustering",
                partitions=store.partition_count,
                max_decoded_leaf_bytes=store.max_leaf_bytes,
                preprocess_mb=preprocess_mb,
                inference="unit scores with first-stage correction; CRV1 is N/(N-1) only",
            )
            if profile:
                return Profile(
                    model="diff-stream gardner",
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
                            "snapshot and sorted partitions",
                            64 * store.n_units * T,
                            128 * store.n_units * T,
                            where="disk",
                        ),
                    ],
                    notes={
                        **diagnostics,
                        "n_units": store.n_units,
                        "periods": T,
                        "coefficients": M,
                    },
                    warnings=[
                        "Heuristic; third-party source operators and allocators are not hard capped."
                    ],
                )
            if not budget.fits(numeric / 1e9 + 0.128):
                raise MemoryError(
                    "estimated numerical buffers exceed job headroom; reduce batch_units or request more memory"
                )
            if iterative:
                from .twostage_absorbed import fit_two_stage_absorbed

                support, horizons, names = second_stage_support(
                    store, specification, reference, anticipation
                )
                beta, V, absorbed_info = fit_two_stage_absorbed(
                    store, fe_terms, specification, horizons, anticipation, threads=threads
                )
                fit_info = {
                    "first_stage_columns": absorbed_info["first_stage_columns"],
                    "absorbed_or_redundant_columns": absorbed_info["absorbed_or_redundant_columns"],
                    "fixed_effect_solver": "iterative",
                    "absorbed_terms": absorbed_info["absorbed_terms"],
                    "regression_backend": absorbed_info["regression_backend"],
                }
                support = pl.DataFrame(support)
            else:
                beta, V, names, support, model = _fit(store, specification, reference, anticipation)
                fit_info = {
                    "first_stage_columns": [model.names[j] for j in model.keep],
                    "absorbed_or_redundant_columns": model.dropped,
                    "fixed_effect_solver": "dense",
                }
            V *= store.n_units / (store.n_units - 1) if vcov == "CRV1" else 1
            V = (V + V.T) / 2
            result = TwoStageResult(
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
                diagnostics={
                    **diagnostics,
                    **fit_info,
                    "warnings": [],
                },
            )
            return present(result, store.period_labels)


@_docs.document
def did_two_stage(
    fml,
    data,
    *,
    panel,
    cohort=None,
    treatment=None,
    event_study=True,
    reference_period=None,
    anticipation=0,
    weights=None,
    allow_time_varying_weights=False,
    incomplete_units="error",
    last_treated_controls="if_needed",
    vcov="CRV0",
    fixed_effect_solver="auto",
    n_threads=None,
    max_memory_gb=None,
    memory_reserve=None,
    workdir=None,
    profile=False,
    batch_units=4096,
    preprocess_mb=None,
):
    """Gardner's two-stage difference-in-differences.

    The first stage fits unit and time fixed effects (and any covariates) on
    the untreated observations only; the second stage regresses the
    first-stage residuals of all observations on treatment indicators. The
    standard errors account for the estimated first stage. The point
    estimates match BJS imputation in many designs; the inference differs.

    Gardner, J. (2022). Two-stage differences in differences. arXiv:2207.05943.

    Butts, K. and Gardner, J. (2022). did2s: two-stage
    difference-in-differences. *The R Journal* 14(3), 162-173.

    Parameters
    ----------
    fml : str
        ``"y ~ 1"``; first-stage covariates ``"y ~ x"``; additional fixed
        effects after a bar, ``"y ~ x | region^time"``. Unit and time fixed
        effects are always included. Fixed effects with many levels
        (county-by-year) are absorbed iteratively; see ``fixed_effect_solver``.
    {DATA}
    event_study : bool, default True
        Estimate one coefficient per event time (an event study). False
        estimates a single overall effect of treatment.
    reference_period : int, optional
        With ``event_study``, the omitted pre-treatment event time. Default:
        the last clean period, ``-anticipation - 1``.
    {ANTICIPATION}
    {WEIGHTS}
    {INCOMPLETE_UNITS}
    {LAST_TREATED}
    {VCOV}
    {FIXED_EFFECT_SOLVER}
    {RESOURCES}

    Returns
    -------
    TwoStageResult or Profile
        The second-stage coefficients (``event::h`` or ``ATT``) and their
        joint covariance. ``select(horizons)`` reports a subset of event
        times without refitting.
    """
    options = dict(locals())
    fml, data = options.pop("fml"), options.pop("data")
    return public_profile(_two_stage(fml, data, **_options.internal(options)))

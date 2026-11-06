"""Gardner two-stage DiD with bounded unit elimination and adjusted inference."""

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

from ._core.nuisance import solve_checked
from ._core.within import moments
from .linear import RowDesign
from .store import PanelStore, inspect


@dataclass(kw_only=True)
class GardnerResult(RegressionResult):
    """Direct second-stage coefficients, rather than cohort-time estimates.

    Selection returns fitted coefficients without changing either regression.
    No temporary store survives estimation.
    """

    n_units: int
    _support: pl.DataFrame = field(repr=False)
    estimator = "gardner"
    cells = ()
    _normal_inference = True

    def support(self):
        """Observed cohort/date counts and original-scale weights per coefficient."""
        return self._support.clone()

    def contributions(self):
        """Support table used by the common pre-treatment Wald helper."""
        return self.support()

    def select(self, horizons):
        """Select fitted dynamic horizons, retaining their full joint covariance."""
        if self.diagnostics["specification"] != "dynamic":
            raise ValueError("horizon selection requires a dynamic Gardner fit")
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


def _fit(store, specification, reference, anticipation):
    model = RowDesign(store)
    model.reduce()
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
    P, M = len(model.keep), len(names)
    Q, q = np.zeros((P, P)), np.zeros(P)
    for g, y, X, w in model.batches():
        clean = (g[:, None] == 0) | (store.times < g[:, None] - anticipation)
        a, b = moments(X, np.nan_to_num(y, nan=0.0), w * clean)
        Q += a
        q += b
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
            Z = Z & obs[:, :, None]
            yield X - xm[:, None], Z, w, u1, score1

    # Unit first-stage scores vanish exactly after clean-sample elimination.
    # The Schur-complement cross derivative uses X - clean-unit-mean(X),
    # even though the second-stage regression itself does not demean Z.
    mass, rhs, C = np.zeros(M), np.zeros(M), np.zeros((M, P))
    for xc, Z, w, u1, _ in batches():
        mass += np.einsum("btm,bt->m", Z, w)
        rhs += np.einsum("btm,bt->m", Z, w * u1)
        C += np.einsum("btm,bt,btk->mk", Z, w, xc, optimize=True)
    if np.any(mass <= 0):
        raise ValueError("Gardner second-stage design is unidentified")
    beta = rhs / mass  # Supported indicators are mutually exclusive; no intercept.
    H = solve_checked(Q, C.T, "Gardner first-stage inference design is unidentified")
    accumulator = ClusterScoreAccumulator()
    for _, Z, w, u1, score1 in batches():
        u2 = u1 - Z @ beta
        score2 = np.einsum("btm,bt->bm", Z, w * u2)
        accumulator.add_clusters((score2 - score1 @ H) / mass)
    return beta, accumulator.sums, names, pl.DataFrame(support), model


def gardner_stream(
    fml,
    data,
    *,
    panel,
    cohort,
    specification="dynamic",
    reference_period=None,
    vcov="CRV0",
    batch_units=4096,
    threads=None,
    memory_gb=None,
    workdir=None,
    profile=False,
    preprocess_mb=64,
    panel_mode="balanced",
    anticipation=0,
    weights=None,
    allow_time_varying_weights=False,
    reference_group="auto",
):
    """Estimate Gardner's two-stage DiD using clean-sample unit/time WLS.

    ``fml`` specifies the outcome and first-stage covariates/additional fixed
    effects, using the same bounded (256 added columns) design as BJS/SA.
    Unit/time intercepts are automatic. ``specification='static'`` estimates
    one indicator for affected observations (t >= g-anticipation). ``'dynamic'``
    fits every observed treated-cohort event time except ``reference_period``
    (default -anticipation-1). There is no second-stage intercept; omitted
    reference and never-treated rows have a zero second-stage design.

    A reference must be a clean integer event time. Use result.select(horizons)
    to report a window without changing either stage. Event times retain actual
    adoption dates. Static ATT includes anticipation periods when requested.
    Covariance uses unit-clustered scores corrected for the estimated first
    stage. CRV1 multiplies CRV0 by N/(N-1) only; normal inference is used.

    Panel, weights, thread/budget and partition options follow bjs_stream.
    reference_group='auto' uses a clean full-window reference if present,
    otherwise retains dates strictly before the last cohort is affected and
    assigns that cohort zero second-stage design, including its leads.
    'nevertreated' is strict; 'lasttreated' explicitly selects that window.
    Requires at least one clean observation per unit and an identified clean
    first-stage design. Returns GardnerResult, or a
    heuristic Profile with profile=True. No refit bootstrap is implemented.
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
        raise ValueError("reference_period must be a clean integer event time for a dynamic fit")
    reference = -anticipation - 1 if reference_period is None else int(reference_period)
    if vcov not in ("CRV0", "CRV1"):
        raise ValueError("vcov must be 'CRV0' or 'CRV1'")
    if panel_mode not in ("balanced", "complete", "unbalanced"):
        raise ValueError("panel_mode must be 'balanced', 'complete' or 'unbalanced'")
    for value, name in [(batch_units, "batch_units"), (threads, "threads")]:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    if batch_units is None:
        raise ValueError("batch_units must be a positive integer")
    if isinstance(preprocess_mb, bool) or not np.isfinite(preprocess_mb) or preprocess_mb < 1:
        raise ValueError("preprocess_mb must be finite and at least 1 MB")
    if not isinstance(allow_time_varying_weights, bool):
        raise ValueError("allow_time_varying_weights must be boolean")
    budget = MemoryBudget(memory_gb)
    baseline = current_rss_gb() or 0
    with thread_scope(threads):
        lf, yname, design = inspect(
            fml, data, panel, cohort, allow_covariates=True, weights=weights, allow_fe=True
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
                aggregation="dynamic" if specification == "dynamic" else "static",
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
                            preprocess_mb * 1e6 + 96e6,
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
            beta, V, names, support, model = _fit(store, specification, reference, anticipation)
            V *= store.n_units / (store.n_units - 1) if vcov == "CRV1" else 1
            V = (V + V.T) / 2
            return GardnerResult(
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
                    "first_stage_columns": [model.names[j] for j in model.keep],
                    "absorbed_or_redundant_columns": model.dropped,
                    "warnings": [],
                },
            )

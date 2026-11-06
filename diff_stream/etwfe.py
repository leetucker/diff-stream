"""Linear extended TWFE with bounded saturated designs and streamed margins.

Raw covariates span the same regression as cohort-demeaned interaction coding;
weighted margin gradients recover effects without depending on the coding basis.
"""

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
from ._core.nuisance import solve_checked
from ._core.partition import preprocess_budget_mb
from ._core.within import center_groups, independent_columns
from ._present import present, public_profile
from .results import DIDResult
from .store import PanelStore, inspect


class ExtendedDesign:
    """Cohort/time slopes and cell-specific slopes, with bounded row batches."""

    def __init__(self, store, control_group, fixed_effects, anticipation):
        self.store, self.fixed_effects = store, fixed_effects
        self.groups = sorted(store.gc)
        self.K = len(store.design.names) if store.design is not None else 0
        self.cells = [
            (g, int(t))
            for g in self.groups
            if g
            for t, n in zip(store.times, store.cell_counts[g], strict=True)
            if n
            and (
                t >= g - anticipation
                if control_group == "notyettreated"
                else t != g - anticipation - 1
            )
        ]
        xnames = store.design.names if store.design is not None else []
        self.names = [] if fixed_effects == "unit" else [f"cohort::{g}" for g in self.groups]
        self.names += [f"time::{t}" for t in store.times[1:]]
        self.names += [f"cohort::{g}:{name}" for g in self.groups for name in xnames]
        self.names += [f"time::{t}:{name}" for t in store.times[1:] for name in xnames]
        self.offset = len(self.names)
        self.names += [f"cell::{g}::{t}:{name}" for g, t in self.cells for name in ["1", *xnames]]
        self.time = np.eye(store.periods)[:, 1:]
        # cell_at[cohort index, time index] = estimated cell number, or -1
        self.group_array = np.asarray(self.groups)
        self.cell_at = np.full((len(self.groups), store.periods), -1, dtype=np.int64)
        t0 = int(store.times[0])
        for j, (cg, t) in enumerate(self.cells):
            self.cell_at[self.groups.index(cg), t - t0] = j

    def batches(self):
        for g, y, x, w in self.store.batches(
            include_x=True, include_weights=True, row_weights=True
        ):
            x = x if self.K else x[:, :, :0]
            x = np.nan_to_num(x, nan=0)
            B, T = y.shape
            D = np.zeros((B, T, len(self.names)))
            at = 0
            if self.fixed_effects == "cohort":
                D[:, :, : len(self.groups)] = (g[:, None] == np.asarray(self.groups))[:, None, :]
                at += len(self.groups)
            D[:, :, at : at + T - 1] = self.time
            at += T - 1
            for cg in self.groups:
                D[:, :, at : at + self.K] = x * (g == cg)[:, None, None]
                at += self.K
            for ti in range(1, T):
                D[:, ti, at : at + self.K] = x[:, ti]
                at += self.K
            one_x = np.concatenate([np.ones((B, T, 1)), x], axis=2)
            # Each unit-period belongs to at most one estimated cell; set every
            # cell's slope block in one scatter instead of a mask per cell.
            cell = self.cell_at[np.searchsorted(self.group_array, g)]
            rows, periods = np.nonzero(cell >= 0)
            index = cell[rows, periods]
            K1 = self.K + 1
            columns = self.offset + index[:, None] * K1 + np.arange(K1)
            D[rows[:, None], periods[:, None], columns] = one_x[rows, periods]
            yield g, np.nan_to_num(y, nan=0), D, w, one_x, (rows, periods, index)


def _fit(model, inference):
    store = model.store
    P, M = len(model.names), len(model.cells)
    Q, q, norms = np.zeros((P, P)), np.zeros(P), np.zeros(P)
    mass = np.array([store.cell_weights[g][t - int(store.times[0])] for g, t in model.cells])
    means = np.zeros((M, model.K + 1))
    for _g, y, D, w, one_x, (rows, periods, index) in model.batches():
        ww = w.ravel()
        if model.fixed_effects == "unit":
            flat = D.reshape(-1, P)
            norms += (flat * flat).T @ ww
            D, y, _, _ = center_groups(D, y, w)
        flat = D.reshape(-1, P)
        weighted = ww[:, None] * flat
        Q += flat.T @ weighted
        q += weighted.T @ y.ravel()
        contribution = w[rows, periods][:, None] * one_x[rows, periods]
        for k in range(model.K + 1):
            means[:, k] += np.bincount(index, weights=contribution[:, k], minlength=M)
    if model.fixed_effects != "unit":
        # Without centering, the column norms are the Gram diagonal.
        norms = np.diag(Q).copy()
    means /= mass[:, None]
    keep = independent_columns(Q, original_norm=norms)
    if not len(keep):
        raise ValueError("ETWFE regression has no identified columns")
    L = np.zeros((M, P))
    for j in range(M):
        L[j, model.offset + j * (model.K + 1) : model.offset + (j + 1) * (model.K + 1)] = means[j]
    # Individual interaction aliases are permissible only if every reported
    # margin is invariant to their normalization. Check in scaled coordinates.
    scale = np.sqrt(np.maximum(np.diag(Q), 0))
    safe = np.where(scale > 0, scale, 1)
    normalized = Q / safe[:, None] / safe[None, :]
    relation = solve_checked(
        normalized[np.ix_(keep, keep)], normalized[keep], "ETWFE column identification"
    )
    gradient = L / safe
    residual = gradient - gradient[:, keep] @ relation
    bad = np.max(abs(residual), axis=1) > 1e-8 * np.maximum(np.max(abs(gradient), axis=1), 1e-12)
    if bad.any():
        raise ValueError(
            f"ETWFE treatment effects are unidentified for cells {[model.cells[j] for j in np.flatnonzero(bad)]}; check reference support and covariate variation"
        )
    Q = Q[np.ix_(keep, keep)]
    coef = solve_checked(Q, q[keep], "ETWFE saturated regression")
    beta = L[:, keep] @ coef
    H = solve_checked(Q, L[:, keep].T, "ETWFE margin covariance")
    full_coef = np.zeros(P)
    full_coef[keep] = coef
    acc = ClusterScoreAccumulator()
    blocks = full_coef[model.offset :].reshape(M, model.K + 1)
    all_columns = len(keep) == P
    for g, y, D, w, one_x, (rows, periods, index) in model.batches():
        if model.fixed_effects == "unit":
            D, y, _, _ = center_groups(D, y, w)
        if not all_columns:
            D = D[:, :, keep]
        e = y - D @ coef
        score = np.matmul((w * e)[:, None, :], D)[:, 0, :] @ H
        if inference == "unconditional":
            share = np.zeros((len(g), M))
            # Recover each row's margin from the retained coefficient coding;
            # its sample average equals the reported cell effect. A unit has
            # one row per cell, so the (unit, cell) pairs are distinct.
            margin = np.einsum("rk,rk->r", one_x[rows, periods], blocks[index])
            row_weight = w[rows, periods]
            share[rows, index] = row_weight
            score[rows, index] += row_weight * (margin - beta[index]) / mass[index]
            acc.add_clusters(score, share - mass / store.n_units)
        else:
            acc.add_clusters(score)
    joint = acc.sums
    return (
        beta,
        joint[:M, :M],
        joint[:M, M:] if inference == "unconditional" else None,
        joint[M:, M:] if inference == "unconditional" else None,
        keep,
        coef,
    )


def _extended_twfe(
    fml,
    data,
    *,
    panel,
    cohort=None,
    treatment=None,
    control_group="nevertreated",
    fixed_effects="cohort",
    inference="unconditional",
    vcov="CRV0",
    weights=None,
    allow_time_varying_weights=False,
    panel_mode="balanced",
    anticipation=0,
    batch_units=None,
    threads=None,
    memory_gb=None,
    memory_reserve=None,
    preprocess_mb=None,
    workdir=None,
    profile=False,
    max_design_columns=2048,
    reference_group="auto",
):
    """Linear extended TWFE from internal option names; see did_extended_twfe.

    Saturates covariates with cohort, time and treatment-cell interactions.
    Never-treated controls also estimate leads, omitting g-anticipation-1;
    not-yet-treated controls estimate affected cells only. Unconditional
    inference includes covariate-distribution and aggregation-share
    variation; conditional holds both fixed.
    """
    if control_group not in ("nevertreated", "notyettreated"):
        raise ValueError("control_group must be nevertreated or notyettreated")
    if reference_group not in ("auto", "nevertreated", "lasttreated"):
        raise ValueError("reference_group must be auto, nevertreated or lasttreated")
    if fixed_effects not in ("cohort", "unit"):
        raise ValueError("fixed_effects must be cohort or unit")
    if inference not in ("conditional", "unconditional"):
        raise ValueError("inference must be conditional or unconditional")
    if vcov not in ("CRV0", "CRV1"):
        raise ValueError("vcov must be CRV0 or CRV1")
    if panel_mode not in ("balanced", "complete", "unbalanced"):
        raise ValueError("panel_mode must be balanced, complete or unbalanced")
    if (
        isinstance(anticipation, (bool, np.bool_))
        or not isinstance(anticipation, Integral)
        or not 0 <= anticipation < np.iinfo(np.int64).max
    ):
        raise ValueError("anticipation must be a nonnegative integer within int64 range")
    anticipation = int(anticipation)
    checked = [(max_design_columns, "max_design_columns")]
    if batch_units is not None:
        checked.append((batch_units, "batch_units"))
    for value, name in checked:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if threads is not None and (
        isinstance(threads, bool) or not isinstance(threads, int) or threads < 1
    ):
        raise ValueError("n_threads must be a positive integer")
    if not isinstance(allow_time_varying_weights, bool):
        raise ValueError("allow_time_varying_weights must be boolean")
    budget = MemoryBudget(memory_gb, memory_reserve)
    preprocess_mb = preprocess_budget_mb(preprocess_mb, budget)
    baseline = current_rss_gb() or 0
    with thread_scope(threads):
        lf, yname, design = inspect(
            fml, data, panel, cohort, treatment=treatment, allow_covariates=True, weights=weights
        )
        if not budget.fits((preprocess_mb * 1e6 + 96e6) / 1e9):
            raise MemoryError(
                "preprocessing buffers exceed job headroom before staging; reduce preprocess_mb or request more memory"
            )
        if workdir is not None:
            Path(workdir).mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="diff-stream-etwfe-", dir=workdir) as run:
            store = PanelStore(
                lf,
                run,
                batch_units or 4096,
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
            K = len(design.names) if design is not None else 0
            cells = [
                (g, int(t))
                for g in sorted(store.gc)
                if g
                for t, n in zip(store.times, store.cell_counts[g], strict=True)
                if n
                and (
                    t >= g - anticipation
                    if control_group == "notyettreated"
                    else t != g - anticipation - 1
                )
            ]
            G, T, M = len(store.gc), store.periods, len(cells)
            P = (G if fixed_effects == "cohort" else 0) + T - 1 + K * (G + T - 1) + M * (K + 1)
            if not M:
                raise ValueError("no supported ETWFE effects")
            if P > max_design_columns:
                raise ValueError(
                    f"ETWFE expanded design requires {P} columns, exceeding max_design_columns={max_design_columns}"
                )

            def buffers(B):
                return 8 * (
                    12 * P * P + 12 * M * M + 8 * M * P + 10 * B * T * (P + K + T) + 8 * B * M
                )

            # An automatic batch starts at 4096 units and halves until it fits.
            B = min(batch_units or 4096, store.n_units)
            if batch_units is None:
                while B > 128 and not budget.fits(buffers(B) / 1e9 + 0.128):
                    B //= 2
            numeric = buffers(B)
            store.batch_units = B
            diagnostics = dict(
                **reference_metadata,
                control_group=control_group,
                fixed_effects=fixed_effects,
                inference=inference,
                anticipation=anticipation,
                reference_event_time=-anticipation - 1,
                base_period="cohort clean pre-period"
                if control_group == "nevertreated"
                else "common time effects from unaffected observations",
                panel_mode=panel_mode,
                balanced_sample=store.is_balanced,
                n_units_input=store.n_units_input,
                dropped_units=store.n_units_input - store.n_units,
                cohort_sizes={str(g): int(n) for g, n in store.gc.items()},
                time_min=int(store.times[0]),
                time_max=int(store.times[-1]),
                sampling_weights=weights,
                weights_time_invariant=store.weights_time_invariant,
                weight_scale=store.weight_scale / store.weight_normalizer,
                expanded_columns=P,
                max_design_columns=max_design_columns,
                partitions=store.partition_count,
                max_decoded_leaf_bytes=store.max_leaf_bytes,
                preprocess_mb=preprocess_mb,
                aggregation_weight="observed target-cell sampling weight totals",
                warnings=[],
            )
            if profile:
                return Profile(
                    model="diff-stream etwfe",
                    threads=threads or detect_cpus(),
                    memory_limit_gb=budget.limit_gb,
                    memory_reserve=budget.reserve,
                    polars_threads=pl.thread_pool_size(),
                    baseline_gb=baseline,
                    components=[
                        Component(
                            "saturated numerical design", numeric + 192e6, numeric * 1.5 + 256e6
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
                    notes={**diagnostics, "n_units": store.n_units, "periods": T, "effects": M},
                    warnings=[
                        "Heuristic; third-party source execution and allocators are not hard capped."
                    ],
                )
            if not budget.fits(numeric / 1e9 + 0.128):
                raise MemoryError(
                    "estimated ETWFE design buffers exceed job headroom; reduce batch_units or request more memory"
                )
            model = ExtendedDesign(store, control_group, fixed_effects, anticipation)
            beta, V, cross, share_cov, keep, coef = _fit(model, inference)
            factor = store.n_units / (store.n_units - 1) if vcov == "CRV1" else 1
            V = (V + V.T) / 2 * factor
            result = DIDResult(
                fml=fml,
                depvar=yname,
                coefnames=[f"ATT::{g}::{t}" for g, t in cells],
                beta=beta,
                vcov=V,
                vcov_type=vcov,
                df_t=store.n_units - 1,
                n_obs=store.n_rows_used,
                n_units=store.n_units,
                estimator="etwfe",
                cells=cells,
                counts=np.array([store.cell_weights[g][t - int(store.times[0])] for g, t in cells]),
                observed_counts=np.array(
                    [store.cell_counts[g][t - int(store.times[0])] for g, t in cells]
                ),
                cohort_counts=np.array([store.gc[g] for g, t in cells]),
                weights=weights,
                n_clusters={panel[0]: store.n_units},
                all_vcovs={vcov: (V, store.n_units - 1)},
                _share_cross=cross * factor if cross is not None else None,
                _share_cov=share_cov * factor if share_cov is not None else None,
                diagnostics={
                    **diagnostics,
                    "regression_columns": [model.names[j] for j in keep],
                    "absorbed_or_redundant_columns": [
                        name for j, name in enumerate(model.names) if j not in keep
                    ],
                    "regression_coefficients": coef.tolist(),
                },
            )
            return present(result, store.period_labels)


@_docs.document
def did_extended_twfe(
    fml,
    data,
    *,
    panel,
    cohort=None,
    treatment=None,
    control_group="never_treated",
    fixed_effects="cohort",
    treat_sample_as_fixed=False,
    anticipation=0,
    weights=None,
    allow_time_varying_weights=False,
    incomplete_units="error",
    last_treated_controls="if_needed",
    vcov="CRV0",
    max_design_columns=2048,
    n_threads=None,
    max_memory_gb=None,
    memory_reserve=None,
    workdir=None,
    profile=False,
    batch_units=None,
    preprocess_mb=None,
):
    """Wooldridge's extended two-way fixed effects (linear).

    A single regression with a separate treatment effect for every cohort and
    period, and covariates interacted with cohort, time and treatment, so
    that the effects can differ with the covariates. The cohort-time effects
    are averaged over the sample's covariates and aggregated with
    ``aggregate()``, as in R's etwfe and emfx. Only the linear model is
    implemented; for logit or Poisson ETWFE use R's etwfe or Stata's jwdid.

    Wooldridge, J. M. (2021). Two-way fixed effects, the two-way Mundlak
    regression, and difference-in-differences estimators. Working paper,
    SSRN 3906345.

    Parameters
    ----------
    fml : str
        ``"y ~ 1"`` or ``"y ~ x + z"``. Additional fixed effects are not
        supported.
    {DATA}
    control_group : {"never_treated", "not_yet_treated"}, default "never_treated"
        The comparison units. With never-treated controls, pre-treatment
        effects are estimated too (for a pre-trend test). R's etwfe defaults
        to not-yet-treated.
    fixed_effects : {"cohort", "unit"}, default "cohort"
        Cohort intercepts (R etwfe's default) or unit fixed effects.
    treat_sample_as_fixed : bool, default False
        Standard errors for the effect in these units, holding the observed
        covariates and cohort shares fixed (smaller), instead of in the
        population they were drawn from (R emfx's conditional option).
    {ANTICIPATION}
    {WEIGHTS}
    {INCOMPLETE_UNITS}
    {LAST_TREATED}
    {VCOV}
    max_design_columns : int, default 2048
        The limit on the expanded design, after interacting the covariates.
    {RESOURCES}

    Returns
    -------
    DIDResult or Profile
        ``cohort_time()`` lists the cohort-time effects; ``aggregate()``
        gives an event study, ``aggregate("overall")`` the overall ATT.
    """
    options = dict(locals())
    fml, data = options.pop("fml"), options.pop("data")
    return public_profile(_extended_twfe(fml, data, **_options.internal(options)))

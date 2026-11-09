"""Published feTR treatment-effect weights for an observed-cell TWFE regression."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from hdfe_stream.core.profile import Component, Profile
from stream_limits import MemoryBudget, current_rss_gb, detect_cpus, thread_scope

from . import _docs, _options
from ._core.nuisance import solve_checked
from ._core.partition import preprocess_budget_mb, staging_bytes
from ._core.within import center_groups, independent_columns
from ._present import public_profile
from .twfe_store import WeightStore, inspect_weights


@dataclass(kw_only=True)
class TWFEWeightResult:
    """Descriptive feTR weights and sensitivity summaries; no ATT inference.

    Cell storage is optional (keep_cells=True). Use cleanup() or a context
    manager to release retained temporary files. Reduced summaries survive it.
    """

    fml: str
    twfe_estimate: float
    n_units: int
    n_obs: int
    diagnostics: dict
    _summary: dict = field(repr=False)
    _sensitivity: dict = field(repr=False)
    _run: object = field(default=None, repr=False)
    _path: Path | None = field(default=None, repr=False)

    def summary(self):
        """Reduced sign counts/sums, weight extrema and population shares."""
        return pl.DataFrame(self._summary["by_sign"])

    def sensitivity(self):
        """Two published sensitivity thresholds with explicit status/conventions."""
        return pl.DataFrame(self._sensitivity["measures"])

    def summary_dict(self):
        return dict(
            model="TWFE treatment-effect weights (feTR)",
            fml=self.fml,
            twfe_estimate=self.twfe_estimate,
            n_units=self.n_units,
            n_obs=self.n_obs,
            diagnostics=self.diagnostics,
            weights=json.loads(json.dumps(self._summary)),
            sensitivity=json.loads(json.dumps(self._sensitivity)),
        )

    def summary_json(self, path=None):
        content = json.dumps(self.summary_dict(), indent=2, allow_nan=False)
        if path is not None:
            Path(path).write_text(content + "\n")
        return content

    def _require_cells(self):
        if self._path is None or not self._path.exists():
            raise ValueError(
                "cell storage is unavailable; fit with keep_cells=True and export before cleanup()"
            )
        return self._path

    def cells(self):
        """Lazy, disk-backed cell table; valid until cleanup()."""
        return pl.scan_parquet(self._require_cells(), low_memory=True)

    def iter_cells(self, batch_rows=32768):
        """Bounded cell frames (ordering across partitions is unspecified)."""
        if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows < 1:
            raise ValueError("batch_rows must be a positive integer")
        with pq.ParquetFile(self._require_cells(), pre_buffer=False) as pf:
            for b in pf.iter_batches(batch_size=batch_rows):
                yield pl.from_arrow(b)

    def write_cells(self, path):
        """Stream cells to a caller-owned Parquet file that survives cleanup()."""
        path = Path(path)
        src = self._require_cells()
        if path.resolve() == src.resolve() or src.parent.resolve() in path.resolve().parents:
            raise ValueError("export cells outside the temporary fit directory")
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = None
        try:
            for b in self.iter_cells():
                table = b.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                writer.write_table(table, row_group_size=32768)
        except BaseException:
            if writer is not None:
                writer.close()
                writer = None
            path.unlink(missing_ok=True)
            raise
        finally:
            if writer is not None:
                writer.close()
        return path

    def cleanup(self):
        if self._run is not None:
            self._run.cleanup()
            self._run = None
        self._path = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()


def _opposite_threshold(path, beta, max_multiplier, zero_tol):
    """Solve the paper's sorted-tail cutoff with bounded repeated reductions.

    For cells with nonzero reference-classified q and W<c, accumulate P,S,T.
    The optimal tail satisfies c=-S/(1-P). Active-set Newton steps converge
    monotonically from c=0; safeguarded bisection avoids a global cell sort.
    """
    lo, hi, cutoff = 0.0, max_multiplier, 0.0
    last = None
    for iteration in range(100):
        P = S = T = 0.0
        with pq.ParquetFile(path, pre_buffer=False) as pf:
            for b in pf.iter_batches(
                batch_size=32768, columns=["relative_weight", "treated_share", "weight"]
            ):
                f = pl.from_arrow(b)
                W, p, q = [f[c].to_numpy() for c in f.columns]
                keep = (np.abs(q) >= zero_tol) & (q != 0) & (W < cutoff)
                P += float(p[keep].sum())
                S += float(q[keep].sum())
                T += float(np.sum(p[keep] * W[keep] ** 2))
        remaining = 1 - P
        if remaining <= 0:
            hi = cutoff
            cutoff = (lo + hi) / 2
            continue
        next_cutoff = -S / remaining
        if last == (P, S, T) or abs(next_cutoff - cutoff) <= 1e-12 * max(1, abs(cutoff)):
            variance = T + S * S / remaining
            if variance > 0 and np.isfinite(variance):
                return abs(beta) / np.sqrt(variance), iteration + 1
            break
        last = (P, S, T)
        if cutoff * remaining + S < 0:
            lo = cutoff
        else:
            hi = cutoff
        cutoff = next_cutoff if lo < next_cutoff < hi else (lo + hi) / 2
    raise FloatingPointError(
        "opposite-sign sensitivity cutoff did not converge; rescale weights/covariates"
    )


def _fit(store, path, zero_tol):
    T = store.periods
    names = [f"time::{t}" for t in store.times[1:]] + (
        [] if store.plan is None else [n for n in store.plan.names if n != "Intercept"]
    )
    P = len(names)
    Q, rhs, norms = np.zeros((P, P)), np.zeros(P), np.zeros(P)
    dd = treated_mass = total_mass = 0.0
    for _, X, D, _Y, w, _, _ in store.matrices():
        xc, dc, _, _ = center_groups(X, D, w)
        z, sw = xc.reshape(-1, P), w.ravel()
        Q += z.T @ (sw[:, None] * z)
        rhs += z.T @ (sw * dc.ravel())
        norms += np.einsum("btp,bt,btp->p", X, w, X, optimize=True)
        dd += float(np.sum(w * dc * dc))
        treated_mass += float(np.sum(w * D))
        total_mass += float(w.sum())
    keep = independent_columns(Q, original_norm=norms)
    coef = np.zeros(P)
    if len(keep):
        coef[keep] = solve_checked(Q[np.ix_(keep, keep)], rhs[keep], "TWFE treatment projection")
    rr = rd = ry = 0.0
    orth = np.zeros(P)
    for _, X, D, Y, w, _, _ in store.matrices():
        xc, dc, _, _ = center_groups(X, D, w)
        _, yc, _, _ = center_groups(X, Y, w)
        r = dc - xc @ coef
        rr += float(np.sum(w * r * r))
        rd += float(np.sum(w * r * dc))
        ry += float(np.sum(w * r * yc))
        orth += np.einsum("btp,bt,bt->p", xc, w, r, optimize=True)
    if not np.isfinite(rr) or rr <= 1e-12 * max(dd, 1e-300) or treated_mass <= 0:
        raise ValueError(
            "treatment is collinear with unit/time effects and covariates; no identified TWFE diagnostic"
        )
    orth_error = float(np.max(np.abs(orth) / np.sqrt(np.maximum(np.diag(Q) * rr, 1e-300))))
    if not np.isclose(rd, rr, rtol=1e-7, atol=1e-14 * dd) or orth_error > 1e-7:
        raise FloatingPointError(
            "TWFE treatment projection lost orthogonality; rescale covariates/weights"
        )
    beta = ry / rr
    if not np.isfinite(beta):
        raise FloatingPointError("TWFE coefficient overflowed; rescale the outcome")
    rows = {
        s: dict(sign=s, n_cells=0, weight_sum=0.0, treated_share=0.0)
        for s in ["positive", "negative", "zero"]
    }
    sums = dict(weight_sum=0.0, outcome_reconstruction=0.0, multiplier_variance=0.0)
    count = 0
    extrema = [np.inf, -np.inf, -np.inf]
    writer = None
    pending, pending_rows = [], 0
    try:
        for frame, X, D, Y, w, u, t in store.matrices():
            xc, dc, _, _ = center_groups(X, D, w)
            _, yc, _, _ = center_groups(X, Y, w)
            r = (dc - xc @ coef)[u, t]
            m, d = w[u, t], D[u, t]
            loading = m * r / rr
            q, p = loading * d, m * d / treated_mass
            W = r * treated_mass / rr
            if not np.isfinite(W).all() or not np.isfinite(q).all():
                raise FloatingPointError("TWFE cell weights overflowed; rescale weights/covariates")
            classified = np.where(np.abs(q) < zero_tol, 0.0, q)
            active = d == 1
            count += int(active.sum())
            for sign, mask in [
                ("positive", classified > 0),
                ("negative", classified < 0),
                ("zero", classified == 0),
            ]:
                mask &= active
                rows[sign]["n_cells"] += int(mask.sum())
                rows[sign]["weight_sum"] += float(classified[mask].sum())
                rows[sign]["treated_share"] += float(p[mask].sum())
            if active.any():
                extrema[0] = min(extrema[0], float(q[active].min()))
                extrema[1] = max(extrema[1], float(q[active].max()))
                extrema[2] = max(extrema[2], float(W[active].max()))
            sums["weight_sum"] += float(q.sum())
            sums["outcome_reconstruction"] += float(loading @ yc[u, t])
            sums["multiplier_variance"] += float(np.sum(p * (W - 1) ** 2))
            output = frame.select(
                pl.col("id").alias("unit"),
                pl.col("t").alias("time"),
                pl.col("d").alias("treatment"),
                pl.col("w").alias("sampling_weight"),
            )
            output = output.with_columns(
                pl.Series("treatment_residual", r),
                pl.Series("outcome_loading", loading),
                pl.Series("weight", q),
                pl.Series("treated_share", p),
                pl.Series("relative_weight", np.where(active, W, np.nan)).fill_nan(None),
                pl.Series("classified_weight", classified),
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    path, output.to_arrow().schema, compression="zstd", use_dictionary=False
                )
            # Small unit batches must not create one Parquet row group each:
            # all-row-group reader metadata would then grow unnecessarily fast.
            pending.append(output)
            pending_rows += len(output)
            if pending_rows >= 32768:
                writer.write_table(pl.concat(pending).to_arrow(), row_group_size=32768)
                pending, pending_rows = [], 0
        if pending:
            writer.write_table(pl.concat(pending).to_arrow(), row_group_size=32768)
    finally:
        if writer is not None:
            writer.close()
    if not all(np.isfinite(v) for v in sums.values()):
        raise FloatingPointError(
            "TWFE sensitivity moments overflowed; inspect weight concentration"
        )
    if not np.isclose(sums["weight_sum"], 1, atol=1e-8, rtol=0):
        raise FloatingPointError("TWFE treatment-effect weights fail normalization")
    summary = dict(
        by_sign=list(rows.values()),
        raw_weight_sum=sums["weight_sum"],
        negative_weight_mass=-rows["negative"]["weight_sum"],
        min_weight=extrema[0],
        max_weight=extrema[1],
        n_treated_cells=count,
        zero_tolerance=zero_tol,
    )
    var = sums["multiplier_variance"]
    measures = []
    if var > 1e-24 and count > 1:
        value = abs(beta) / np.sqrt(var * count / (count - 1))
        status = "zero_coefficient" if beta == 0 else "finite"
    else:
        value, status = (
            None,
            "undefined_zero_coefficient" if beta == 0 else "infinite_constant_multiplier",
        )
    measures.append(dict(measure="zero_average_sd", value=value, status=status))
    iterations = 0
    if beta == 0:
        value, status = None, "undefined_zero_coefficient"
    elif rows["negative"]["n_cells"]:
        value, iterations = _opposite_threshold(path, beta, extrema[2], zero_tol)
        status = "finite"
    else:
        value, status = None, "infeasible_no_negative_weights"
    measures.append(dict(measure="all_opposite_sign_sd", value=value, status=status))
    sensitivity = dict(
        measures=measures,
        multiplier_population_variance=var,
        zero_average_sd_convention="R TwoWayFEWeights: weighted SD times sqrt(M/(M-1)), M=treated cell count",
        opposite_sign_algorithm="bounded-pass tail-cutoff reduction",
        cutoff_iterations=iterations,
    )
    meta = dict(
        projection_columns=[names[j] for j in keep],
        absorbed_or_redundant_columns=[n for j, n in enumerate(names) if j not in keep],
        date_components=T - len(independent_columns(Q[: T - 1, : T - 1])),
        projection_orthogonality_error=orth_error,
        residualized_treatment_variance=rr / total_mass,
        outcome_loading_reconstruction_error=sums["outcome_reconstruction"] - beta,
    )
    return float(beta), summary, sensitivity, meta


def _twfe_weights(
    fml,
    data,
    *,
    panel,
    treatment=None,
    cohort=None,
    panel_mode="balanced",
    weights=None,
    allow_time_varying_weights=False,
    keep_cells=False,
    zero_tol=1e-10,
    batch_units=256,
    preprocess_mb=None,
    threads=None,
    memory_gb=None,
    memory_reserve=None,
    workdir=None,
    profile=False,
    max_design_columns=2048,
):
    """TWFE treatment-effect weights from internal option names; see twfe_weights.

    Native unbalanced fits use observed cells. zero_tol classifies summary
    weights as zero (R's 1e-10 default) without changing raw cell weights.
    keep_cells retains temporary Parquet until cleanup().
    """
    if panel_mode not in ("balanced", "unbalanced", "complete"):
        raise ValueError("panel_mode must be balanced, unbalanced or complete")
    for name, value in [("batch_units", batch_units), ("max_design_columns", max_design_columns)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if threads is not None and (
        isinstance(threads, bool) or not isinstance(threads, int) or threads < 1
    ):
        raise ValueError("n_threads must be a positive integer")
    if isinstance(zero_tol, bool) or not np.isfinite(zero_tol) or zero_tol < 0:
        raise ValueError("zero_tolerance must be finite and nonnegative")
    budget = MemoryBudget(memory_gb, memory_reserve)
    preprocess_mb = preprocess_budget_mb(preprocess_mb, budget)
    baseline = current_rss_gb() or 0
    lf, y, plan = inspect_weights(fml, data, panel, treatment, cohort, weights)
    if workdir is not None:
        Path(workdir).mkdir(parents=True, exist_ok=True)
    run = TemporaryDirectory(prefix="diff-stream-twfe-weights-", dir=workdir)
    try:
        with thread_scope(threads):
            store = WeightStore(
                lf,
                run.name,
                budget,
                preprocess_mb,
                batch_units,
                plan,
                panel_mode,
                allow_time_varying_weights,
            )
            K = 0 if plan is None else sum(n != "Intercept" for n in plan.names)
            P, T, B = store.periods - 1 + K, store.periods, min(batch_units, store.n_units)
            numeric = 8 * (10 * P * P + 12 * B * T * (P + 4)) + 48e6
            meta = dict(
                type="feTR",
                panel_mode=panel_mode,
                treatment=treatment,
                cohort=cohort,
                sampling_weights=weights,
                weights_time_invariant=store.weights_time_invariant,
                periods=T,
                input_rows=store.input_rows,
                input_units=store.input_units,
                dropped_rows=store.input_rows - store.n_obs,
                dropped_units=store.input_units - store.n_units,
                partitions=store.partition_count,
                max_decoded_leaf_bytes=store.max_leaf_bytes,
                design_columns=P,
                batch_units=batch_units,
                numerical_bytes=numeric,
                covariate_names=[] if plan is None else [n for n in plan.names if n != "Intercept"],
                treatment_effect_assumptions="common trends and contemporaneous binary treatment effects; no stable-effect restriction",
                zero_tolerance=zero_tol,
                keep_cells=keep_cells,
            )
            if profile:
                return Profile(
                    model="diff-stream TWFE treatment-effect weights",
                    threads=threads or detect_cpus(),
                    memory_limit_gb=budget.limit_gb,
                    memory_reserve=budget.reserve,
                    polars_threads=pl.thread_pool_size(),
                    baseline_gb=baseline,
                    components=[
                        Component(
                            "bounded treatment projection and cell reductions",
                            numeric,
                            1.5 * numeric,
                        ),
                        Component(
                            "bounded partition preprocessing",
                            staging_bytes(preprocess_mb, store) + 96e6,
                            preprocess_mb * 1e6 + 256e6,
                        ),
                        Component(
                            "snapshot, partitions and cell weights",
                            store.n_obs * 128,
                            store.n_obs * 256,
                            where="disk",
                        ),
                    ],
                    notes=meta,
                    warnings=[
                        "Heuristic; dense state grows with dates/covariates and batch size. Cell output scales on disk."
                    ],
                )
            if P > max_design_columns:
                raise ValueError(
                    "TWFE projection exceeds max_design_columns; reduce dates/covariates or explicitly raise the guard"
                )
            if not budget.fits(numeric / 1e9):
                raise MemoryError(
                    "TWFE projection buffers exceed job headroom; reduce batch_units/dates/covariates"
                )
            path = Path(run.name) / "cells.parquet"
            beta, summary, sensitivity, projection = _fit(store, path, zero_tol)
            meta.update(projection)
            for source_path in store.paths:
                source_path.unlink()
            result = TWFEWeightResult(
                fml=fml,
                twfe_estimate=beta,
                n_units=store.n_units,
                n_obs=store.n_obs,
                diagnostics=meta,
                _summary=summary,
                _sensitivity=sensitivity,
            )
            if keep_cells:
                result._run, result._path = run, path
                run = None
            return result
    finally:
        if run is not None:
            run.cleanup()


@_docs.document
def twfe_weights(
    fml,
    data,
    *,
    panel,
    treatment=None,
    cohort=None,
    weights=None,
    allow_time_varying_weights=False,
    incomplete_units="error",
    keep_cells=False,
    zero_tolerance=1e-10,
    max_design_columns=2048,
    n_threads=None,
    max_memory_gb=None,
    memory_reserve=None,
    workdir=None,
    profile=False,
    batch_units=256,
    preprocess_mb=None,
):
    """de Chaisemartin and D'Haultfoeuille's weights behind a TWFE estimate.

    Under heterogeneous effects, the coefficient on a binary treatment in a
    regression with unit and time fixed effects is a weighted sum of the
    effects in each treated unit-period, and some weights can be negative.
    This reports those weights and two sensitivity measures: how much effect
    heterogeneity would be needed for the estimate to have the opposite sign
    from every unit-period's effect. It is a diagnostic, not an estimator of
    a causal effect. It follows the authors' R package TwoWayFEWeights
    (``type="feTR"``).

    de Chaisemartin, C. and D'Haultfoeuille, X. (2020). Two-way fixed effects
    estimators with heterogeneous treatment effects. *American Economic
    Review* 110(9), 2964-2996.

    Parameters
    ----------
    fml : str
        ``"y ~ 1"``, or ``"y ~ x + z"`` with covariates. Unit and time fixed
        effects are always included.
    data : polars.DataFrame, polars.LazyFrame, str or Path
        The panel, or a Parquet path or glob.
    panel : tuple of str
        The unit and time columns. Times are positive integers; gaps are
        allowed.
    treatment, cohort : str, optional
        Exactly one: a binary treatment column (which may switch on and off),
        or a first-treated-period column (0 = never treated).
    weights : str, optional
        A column of positive sampling weights.
    allow_time_varying_weights : bool, default False
        Accept weights that change over time within a unit.
    incomplete_units : {"error", "drop", "keep"}, default "error"
        Refuse a panel with missing unit-periods, drop the incomplete units,
        or use the observed unit-periods.
    keep_cells : bool, default False
        Keep every treated cell's weight for ``cells()``, ``iter_cells()`` and
        ``write_cells()`` (on disk, until ``cleanup()``).
    zero_tolerance : float, default 1e-10
        Weights smaller than this in absolute value count as zero in the
        summary (R's default).
    max_design_columns : int, default 2048
        The limit on the expanded covariate design.
    {RESOURCES}

    Returns
    -------
    TWFEWeightResult or Profile
        ``summary()`` reports the positive and negative weights and the
        sensitivity measures.
    """
    options = dict(locals())
    fml, data = options.pop("fml"), options.pop("data")
    return public_profile(_twfe_weights(fml, data, **_options.internal(options)))

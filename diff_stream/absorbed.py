"""Sun-Abraham and BJS with the fixed effects absorbed iteratively by hdfe-stream.

Sun-Abraham is one hdfe-stream regression: the cohort-period indicators are a
single categorical, with the reference (never-treated units and every cohort's
base period) omitted.

For BJS, the untreated model (unit, time and additional fixed effects, covariates) is
fitted by hdfe-stream on the untreated rows, and its predictions impute the
treated rows' untreated outcomes. Standard errors are Borusyak, Jaravel and
Spiess's: each unit's score sums v_it * e_it, where v are the imputation
weights and e the residuals (treated residuals net of their averages within
``residual_averaging`` groups).

An untreated row's weight in a cell's estimate is -w_it z_it'u_j, with
u_j = S^- Z_j'v_j the solution of the untreated model's normal equations for
the cell's treated design (``HDFEResult.solve_normal_equations``). A unit's
untreated score is therefore linear in the aggregation weights: it is
computed once per cell, kept on disk next to the residuals, and combined for
any aggregation. Only the treated part, whose residual averages depend on the
aggregation, is recomputed.
"""

import warnings
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl

from ._core.partition import whole_group_batches

_DROPPED = ".*variables dropped due to multicollinearity"


def staged(store, terms, paths=None):
    """The staged rows under the names the hdfe-stream formula uses.

    hdfe-stream reserves names starting with "__", so the staged covariate
    and fixed-effect columns are renamed (``__x0`` to ``x0``, ``__fe0`` to
    ``fe0``).
    """
    covariates = list(store.design.names) if store.design is not None else []
    xs = [f"x{j}" for j in range(len(covariates))]
    effects = sorted({c for term, _ in terms for c in term.split("^")} - {"t"})
    columns = ["u", "t", "g", "y", *(["w"] if store.weighted else [])]
    frame = pl.scan_parquet(paths or store.paths, low_memory=True).select(
        *columns,
        *[pl.col(f"__x{j}").alias(x) for j, x in enumerate(xs)],
        *[pl.col(c).alias(c[2:]) for c in effects],
    )
    absorbed = [term.replace("__fe", "fe") for term, _ in terms]
    return frame, covariates, xs, absorbed


# hdfe-stream's row batch for models with fixed effects is 2,000,000 rows,
# whatever the design's width; a batch holds about four float64 copies of it.
DEFAULT_BATCH_ROWS = 2_000_000
BATCH_BYTES_PER_VALUE = 32

# hdfe-stream solves the normal equations for a block of columns at a time, one
# pass over the rows per block. A block holds about this many bytes per unit and
# column (the grouped sums, their copies and the solution), measured at 3 million
# units: 115 MB for a column.
SOLVE_BYTES_PER_UNIT_COLUMN = 40
SOLVE_BLOCK = 16  # hdfe-stream's default; fewest passes


def solve_block(store, columns):
    """Columns per block for the normal-equations solve: hdfe-stream's default
    when half the memory left holds it, fewer (at least one) when it does not."""
    room = 0.5 * store.budget.headroom_gb() * 1e9
    fit = int(room // (SOLVE_BYTES_PER_UNIT_COLUMN * max(store.n_units, 1)))
    return max(1, min(SOLVE_BLOCK, columns, fit))


def batch_rows(store, width):
    """hdfe-stream's row batch: its default when that fits a quarter of the
    memory left, fewer rows (at least 50,000) when it does not."""
    room = 0.25 * store.budget.headroom_gb() * 1e9
    per_row = BATCH_BYTES_PER_VALUE * (width + 2)
    if DEFAULT_BATCH_ROWS * per_row <= room:
        return None
    return max(50_000, int(room / per_row))


def hdfe_options(store, threads, width, **options):
    """hdfe-stream options: diff-stream's memory budget, threads and weights.

    ``width`` is the number of columns in the regression's design.
    """
    rows = batch_rows(store, width)
    if rows is not None:
        options["batch_rows"] = rows
    options.update(
        verbose=False,
        fixef_rm="none",
        max_memory_gb=store.budget.limit_gb,
        memory_reserve=store.budget.reserve,
    )
    if store.weighted:
        options["weights"] = "w"
    if threads is not None:
        options["n_threads"] = threads
    return options


def untreated(anticipation):
    return (pl.col("g") == 0) | (pl.col("t") < pl.col("g") - anticipation)


def fit_sunab_absorbed(store, cells, terms, *, threads=None):
    """Sun-Abraham's regression; returns CRV0 by unit."""
    from hdfe_stream import feols_stream

    keys = pl.DataFrame(
        {
            "g": [g for g, _ in cells],
            "t": [t for _, t in cells],
            "cell": [f"c{j}" for j in range(len(cells))],
        },
        schema={"g": pl.Int64, "t": pl.Int64, "cell": pl.String},
    )
    frame, covariates, xs, absorbed = staged(store, terms)
    data = (
        frame.with_columns(pl.col("g").cast(pl.Int64), pl.col("t").cast(pl.Int64))
        .join(keys.lazy(), on=["g", "t"], how="left")
        .with_columns(pl.col("cell").fill_null("ref"))
    )
    rhs = " + ".join([*xs, "i(cell, ref='ref')"])
    fml = f"y ~ {rhs} | " + " + ".join(["u", "t", *absorbed])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", _DROPPED)
        fit = feols_stream(
            fml,
            data,
            workdir=store.root,
            **hdfe_options(
                store, threads, len(xs) + len(cells), vcov={"CRV1": "u"}, save_resid=False
            ),
        )
    try:
        names = list(fit.coefnames)
        wanted = [f"cell::c{j}" for j in range(len(cells))]
        missing = [cells[j] for j, n in enumerate(wanted) if n not in names]
        if missing:
            raise ValueError(
                f"Sun-Abraham cohort-period effects {missing} are collinear with the fixed effects"
            )
        ix = [names.index(n) for n in wanted]
        beta = np.asarray(fit.beta, dtype=float)[ix]
        factor = fit.diagnostics["vcov_factors"]["CRV1:u"]
        V = np.asarray(fit.vcov, dtype=float)[np.ix_(ix, ix)] / factor
    finally:
        fit.cleanup()
    counts = np.array([store.cell_weights[g][t - store.times[0]] for g, t in cells])
    return beta, counts, (V + V.T) / 2, info(covariates, xs, names, terms)


def info(covariates, xs, names, terms):
    """Diagnostics of an absorbed fit."""
    dropped = [n for n, x in zip(covariates, xs, strict=True) if x not in names]
    return dict(
        regression_backend="hdfe-stream alternating projections; fixed effects absorbed",
        absorbed_terms=[written for _, written in terms],
        regression_columns=[n for n in covariates if n not in dropped],
        absorbed_or_redundant_columns=dropped,
    )


def fit_bjs_absorbed(
    store, cells, terms, avgeffectsby, leaveout, workdir, anticipation=0, *, threads=None
):
    from hdfe_stream import feols_stream

    frame, covariates, xs, absorbed = staged(store, terms)
    fml = f"y ~ {' + '.join(xs) or '1'} | " + " + ".join(["u", "t", *absorbed])
    options = hdfe_options(store, threads, len(xs), vcov="iid", keep_intermediates=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", _DROPPED)
        first = feols_stream(
            fml, frame.filter(untreated(anticipation)), workdir=store.root, **options
        )
    try:
        context = AbsorbedBJSInference(
            store, cells, first, terms, avgeffectsby, leaveout, workdir, anticipation
        )
    finally:
        first.cleanup()
    try:
        V = context.covariance(np.eye(len(cells)), allow_unidentified=avgeffectsby != "cohort_time")
    except BaseException:
        context.cleanup()
        raise
    return context.beta, context.counts, V, context, info(covariates, xs, first.coefnames, terms)


class AbsorbedBJSInference:
    """Residuals and per-cell untreated unit scores, on disk, for aggregation.

    The staged rows are rewritten with the residual (the outcome less its
    imputed untreated outcome), and a memory-mapped array holds, per unit, the
    untreated score of every cell. ``covariance`` makes two passes over them.
    """

    def __init__(self, store, cells, first, terms, avgeffectsby, leaveout, workdir, anticipation):
        self.workspace = TemporaryDirectory(prefix="diff-stream-bjs-", dir=workdir)
        self.store, self.cells = store, cells
        self.avgeffectsby, self.leaveout = avgeffectsby, leaveout
        self.anticipation = anticipation
        self.closed = False
        M, times = len(cells), store.times
        keys = pl.DataFrame(
            {"g": [g for g, _ in cells], "t": [t for _, t in cells], "cell": range(M)},
            schema={"g": pl.Int64, "t": pl.Int64, "cell": pl.Int64},
        )
        weight = pl.col("w") if store.weighted else pl.lit(1.0)

        # Residuals: the outcome less the imputed untreated outcome, in one
        # streaming pass that keeps the staged (unit, period) order.
        frame, _, xs, absorbed = staged(store, terms)
        residuals = f"{self.workspace.name}/residuals.parquet"
        (
            first.predict(frame, name="prediction")
            .with_columns(pl.col("g").cast(pl.Int64), pl.col("t").cast(pl.Int64))
            .join(keys.lazy(), on=["g", "t"], how="left", maintain_order="left")
            .with_columns(
                (pl.col("y") - pl.col("prediction")).alias("r"),
                untreated(anticipation).alias("untreated"),
            )
            .drop("prediction")
            .sink_parquet(residuals, row_group_size=store.read_rows, engine="streaming")
        )
        unidentified = pl.scan_parquet(residuals).select(pl.col("r").null_count()).collect().item()
        if unidentified:
            raise ValueError(
                f"BJS imputation is not identified for {unidentified:,} treated "
                "observations: a fixed-effect level of theirs has no untreated observations"
            )
        for path in store.paths:
            path.unlink()
        store.paths = self.paths = [residuals]

        cell_rows = pl.scan_parquet(residuals).filter(pl.col("cell").is_not_null())
        sums = (
            cell_rows.group_by("cell")
            .agg(
                (weight * pl.col("r")).sum().alias("effect"),
                (pl.col("w").sum() if store.weighted else pl.len().cast(pl.Float64)).alias("mass"),
            )
            .collect()
            .sort("cell")
        )
        mass = np.zeros(M)
        mass[sums["cell"].to_numpy()] = sums["mass"].to_numpy()
        effect = np.zeros(M)
        effect[sums["cell"].to_numpy()] = sums["effect"].to_numpy()
        self.beta = np.divide(effect, mass, out=np.full(M, np.nan), where=mass > 0)
        self.mass = mass
        self.counts = np.array([store.cell_weights[g][t - times[0]] for g, t in cells])

        # The untreated model's normal equations, solved for every cell's
        # treated design. Untreated weights are -w z'u_j.
        names = [f"v{j}" for j in range(M)]
        target = cell_rows.select(
            "u",
            "t",
            *[c for c in first_columns(first) if c not in ("u", "t")],
            *[
                pl.when(pl.col("cell") == j).then(weight / mass[j]).otherwise(0.0).alias(n)
                for j, n in enumerate(names)
            ],
        )
        solution = first.solve_normal_equations(target, names, block=solve_block(store, M))
        try:
            self._untreated_scores(solution, names, xs, absorbed)
        finally:
            solution.cleanup()

    def _untreated_scores(self, solution, names, xs, absorbed):
        """Per unit and cell: -sum over untreated rows of w e z'u_j.

        The unit effects drop out: the untreated model's normal equation for a
        unit's effect makes its untreated rows' weighted residuals sum to zero.
        """
        store, M = self.store, len(names)
        tables = []
        for term in absorbed_terms(absorbed):
            table = solution.fixef(term).collect()
            keys = [c for c in table.columns if c not in names]
            tables.append((keys, table.with_row_index("__level"), table.select(names).to_numpy()))
        kept = solution.coef["coefficient"].to_list()
        coef = solution.coef.select(names).to_numpy()
        # Staged units are numbered 0, ..., N-1: the unit is the row.
        self.scores = f"{self.workspace.name}/scores.npy"
        scores = np.lib.format.open_memmap(
            self.scores, mode="w+", dtype=np.float64, shape=(store.n_units, M)
        )
        for path in self.paths:
            store._guard(32e6 + 8 * M * 4 * store.batch_units * store.periods)
            for d in whole_group_batches(path, "u", store.batch_units, store.read_rows):
                d = d.filter(pl.col("untreated"))
                if not d.height:
                    continue
                q = np.zeros((d.height, M))
                for keys, table, values in tables:
                    level = (
                        d.select(keys)
                        .with_columns(pl.col(c).cast(table.schema[c]) for c in keys)
                        .join(
                            table.select(*keys, "__level"),
                            on=keys,
                            how="left",
                            maintain_order="left",
                        )["__level"]
                        .to_numpy()
                    )
                    q += values[level]
                if kept:
                    q += d.select(kept).to_numpy() @ coef
                w = d["w"].to_numpy() if store.weighted else np.ones(d.height)
                contribution = -(w * d["r"].to_numpy())[:, None] * q
                u = d["u"].to_numpy()
                starts = np.r_[0, np.flatnonzero(u[1:] != u[:-1]) + 1]
                scores[u[starts]] = np.add.reduceat(contribution, starts, axis=0)
        scores.flush()
        del scores

    def cleanup(self):
        self.workspace.cleanup()
        self.closed = True

    def covariance(self, contrasts, *, allow_unidentified=False):
        if self.closed:
            raise ValueError("BJS aggregation data were removed by cleanup(); refit to aggregate")
        store = self.store
        A = np.atleast_2d(contrasts)
        P, M = A.shape
        T, t0 = store.periods, int(store.times[0])
        labels, groups = {}, {}
        for j, (g, t) in enumerate(self.cells):
            label = {"cohort_time": (g, t), "event_time": t - g, "cohort": g, "all": 0}[
                self.avgeffectsby
            ]
            r = labels.setdefault(label, len(labels))
            groups.setdefault((g, r), []).append(j)
        R = len(labels)
        weights = {key: A[:, js] / self.mass[js] for key, js in groups.items()}
        columns = ["u", "t", "g", "r", *(["w"] if store.weighted else [])]

        def batches():
            untreated = np.load(self.scores, mmap_mode="r")
            for path in self.paths:
                for d in whole_group_batches(path, "u", store.batch_units, store.read_rows):
                    d = d.select(columns)
                    u = d["u"].to_numpy()
                    starts = np.r_[0, np.flatnonzero(u[1:] != u[:-1]) + 1]
                    row = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(u)]))
                    ti = d["t"].to_numpy() - t0
                    resid = np.full((len(starts), T), np.nan)
                    resid[row, ti] = d["r"].to_numpy()
                    sw = np.zeros((len(starts), T))
                    sw[row, ti] = d["w"].to_numpy() if store.weighted else 1.0
                    first = np.asarray(untreated[u[starts]])
                    yield d["g"].to_numpy()[starts], resid, sw, first

        def parts(g, resid, sw):
            observed = np.isfinite(resid)
            resid = np.nan_to_num(resid)
            for (cg, r), js in groups.items():
                mask = g == cg
                if not mask.any():
                    continue
                ti = np.array([self.cells[j][1] - t0 for j in js])
                w = weights[(cg, r)]
                totals = (sw[mask][:, ti] * observed[mask][:, ti]) @ w.T
                effects = (sw[mask][:, ti] * resid[mask][:, ti]) @ w.T
                yield mask, r, totals, effects

        denom, numer = np.zeros((P, R)), np.zeros((P, R))
        for g, resid, sw, _ in batches():
            for _, r, totals, effects in parts(g, resid, sw):
                denom[:, r] += (totals**2).sum(axis=0)
                numer[:, r] += (totals * effects).sum(axis=0)
        averages = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
        V, unidentified = np.zeros((P, P)), np.zeros(P, bool)
        for g, resid, sw, first in batches():
            scores = first @ A.T
            for mask, r, totals, effects in parts(g, resid, sw):
                contribution = effects - totals * averages[:, r]
                if self.leaveout:
                    lev = np.divide(
                        totals**2, denom[:, r], out=np.zeros_like(totals), where=denom[:, r] > 0
                    )
                    invalid = (np.abs(totals) > 0) & (lev >= 1 - 1e-12)
                    unidentified |= invalid.any(axis=0)
                    if invalid.any() and not allow_unidentified:
                        raise ValueError(
                            "BJS leave_out requires more than one contributing unit in each residual group; use a coarser residual_averaging"
                        )
                    contribution = np.divide(
                        contribution,
                        1 - lev,
                        out=np.zeros_like(contribution),
                        where=lev < 1 - 1e-12,
                    )
                scores[mask] += contribution
            V += scores.T @ scores
        V[unidentified, :] = np.nan
        V[:, unidentified] = np.nan
        return (V + V.T) / 2


def absorbed_terms(absorbed):
    """The fixed-effect terms other than the unit's, as hdfe-stream names them."""
    return ["t", *absorbed]


def first_columns(first):
    """The source columns the untreated model uses."""
    est = first._estimator
    columns = {c for d in first.fe_names for c in est.fe_cols[d]}
    for name in first.coefnames:
        columns |= set(est.var_exprs[name].meta.root_names())
    return sorted(columns)


def fit_placebos_absorbed(
    store, pretrends, anticipation, terms, *, fml, depvar, panel, vcov, threads=None
):
    """The BJS pre-trend regression with the fixed effects absorbed.

    The untreated rows (by original cohort), with indicators for the last
    ``pretrend_leads`` periods before treatment, on the untreated model.
    """
    from hdfe_stream import feols_stream

    from .bjs_placebo import BJSPlaceboResult

    L = pretrends
    horizons = np.arange(-anticipation - L, -anticipation, dtype=np.int64)
    support, rows = np.zeros(L, dtype=int), []
    scale = store.weight_scale / store.weight_normalizer
    for g, counts in sorted(store.actual_cell_counts.items()):
        if not g:
            continue
        for ti, t in enumerate(store.times):
            j = int(t) - g + anticipation + L
            if 0 <= j < L and counts[ti]:
                support[j] += counts[ti]
                rows.append(
                    dict(
                        aggregate=f"event::{horizons[j]}",
                        cohort=int(g),
                        time=int(t),
                        event_time=int(horizons[j]),
                        observed_units=int(counts[ti]),
                        weight_total=float(store.actual_cell_weights[g][ti] * scale),
                    )
                )
    if np.any(support == 0):
        raise ValueError(
            f"BJS requested clean placebo horizons are not observed: {horizons[support == 0].tolist()}; reduce pretrend_leads"
        )
    frame, covariates, xs, absorbed = staged(store, terms)
    actual = pl.scan_parquet(store.paths, low_memory=True).select("g_actual")
    event = pl.col("t") - pl.col("g_actual")
    data = (
        pl.concat([frame, actual], how="horizontal")
        .filter((pl.col("g_actual") == 0) | (pl.col("t") < pl.col("g_actual") - anticipation))
        .with_columns(
            pl.when((pl.col("g_actual") != 0) & event.is_in(horizons.tolist()))
            .then(pl.format("e{}", event - int(horizons[0])))
            .otherwise(pl.lit("ref"))
            .alias("lead")
        )
    )
    rhs = " + ".join([*xs, "i(lead, ref='ref')"])
    model = f"y ~ {rhs} | " + " + ".join(["u", "t", *absorbed])
    options = hdfe_options(store, threads, len(xs) + L, vcov={"CRV1": "u"})
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", _DROPPED)
        fit = feols_stream(model, data, workdir=store.root, **options)
    try:
        names = list(fit.coefnames)
        wanted = [f"lead::e{j}" for j in range(L)]
        if any(n not in names for n in wanted):
            raise ValueError("BJS clean placebo design is unidentified; reduce pretrend_leads")
        ix = [names.index(n) for n in wanted]
        V = (
            np.asarray(fit.vcov, dtype=float)[np.ix_(ix, ix)]
            / fit.diagnostics["vcov_factors"]["CRV1:u"]
        )
        V *= store.n_units / (store.n_units - 1) if vcov == "CRV1" else 1.0
        beta = np.asarray(fit.beta, dtype=float)[ix]
        n_obs = int(fit.n_obs)
        kept = [n for n, x in zip(covariates, xs, strict=True) if x in names]
    finally:
        fit.cleanup()
    return BJSPlaceboResult(
        fml=fml,
        depvar=depvar,
        coefnames=[f"event::{h}" for h in horizons],
        beta=beta,
        vcov=(V + V.T) / 2,
        vcov_type=vcov,
        df_t=store.n_units - 1,
        all_vcovs={vcov: ((V + V.T) / 2, store.n_units - 1)},
        n_obs=n_obs,
        n_clusters={panel[0]: store.n_units},
        n_units=store.n_units,
        weights=None,
        _support=pl.DataFrame(rows),
        diagnostics=dict(
            aggregation="event_time",
            anticipation=anticipation,
            reference_event_time=None,
            fitted_horizons=horizons.tolist(),
            pretrends=L,
            base_period="all other clean observations; no fixed event-time reference",
            inference="unit clustered; clean-sample regression with absorbed fixed effects",
            regression_columns=[*kept, *(f"event::{h}" for h in horizons)],
            absorbed_or_redundant_columns=[n for n in covariates if n not in kept],
            absorbed_terms=[written for _, written in terms],
            warnings=[],
        ),
    )


class UnimputedObservationsWarning(UserWarning):
    """BJS dropped treated observations whose untreated outcome cannot be imputed."""


def drop_unimputable(store, terms, anticipation):
    """Drop the treated rows with a fixed-effect level no untreated row has.

    Their untreated outcomes are not identified: the untreated model has no
    effect for that level (a county-year in which every unit is treated, say).
    Dropping them changes no other estimate, but the effects then average the
    remaining treated observations only, so the fit warns and reports the
    affected cells.
    """
    from .reference import restage

    rows = pl.scan_parquet(store.paths, low_memory=True)
    clean = untreated(anticipation)
    flags = []
    for j, (term, _) in enumerate(terms):
        columns = term.split("^")
        seen = (
            rows.filter(clean)
            .select(columns)
            .unique()
            .with_columns(pl.lit(True).alias(f"__seen{j}"))
        )
        rows = rows.join(seen, on=columns, how="left", maintain_order="left")
        flags.append(f"__seen{j}")
    unimputable = ~clean & pl.any_horizontal(pl.col(f).is_null() for f in flags)
    rows = rows.with_columns(unimputable.alias("__unimputable")).drop(flags)
    weight = pl.col("w") if store.weighted else pl.lit(1.0)
    treated = rows.filter(~clean)
    totals, by_cell = pl.collect_all(
        [
            treated.select(
                (weight * pl.col("__unimputable").cast(pl.Float64)).sum().alias("dropped"),
                (weight + 0 * pl.col("y")).sum().alias("treated"),
                pl.col("__unimputable").sum().alias("rows"),
            ),
            treated.group_by("g", "t")
            .agg(
                pl.col("__unimputable").sum().alias("dropped"),
                pl.len().alias("observations"),
            )
            .filter(pl.col("dropped") > 0)
            .sort("g", "t"),
        ]
    )
    n = int(totals["rows"][0])
    if not n:
        return store, dict(unimputed_observations=0)
    share = float(totals["dropped"][0] / totals["treated"][0])
    if n == treated.select(pl.len()).collect().item():
        raise ValueError(
            "BJS is unidentified: every treated observation has a fixed-effect level with no "
            "untreated observations (fixed effects such as cohort-by-period absorb the treatment)"
        )
    cells = [
        dict(
            cohort=int(g),
            time=int(t),
            dropped_observations=int(k),
            remaining_observations=int(m - k),
        )
        for g, t, k, m in by_cell.iter_rows()
    ]
    restricted = restage(
        store,
        rows.filter(~pl.col("__unimputable")).drop("__unimputable"),
        "imputable",
        "unbalanced",
        anticipation,
    )
    for path in store.paths:
        path.unlink()
    labels = store.period_labels
    listed = ", ".join(
        f"{labels.label_cohort(c['cohort'])}/{labels.label_int(c['time'])}" for c in cells[:5]
    )
    more = f" and {len(cells) - 5} more" if len(cells) > 5 else ""
    warnings.warn(
        f"{n:,} treated observations ({share:.1%} of the treated weight) have a fixed-effect "
        "level with no untreated observations, so their untreated outcomes cannot be "
        "imputed. They are dropped, and the effects average the remaining treated "
        f"observations. Cohort/period cells affected: {listed}{more}; see "
        "fit.diagnostics['unimputed_cells'].",
        UnimputedObservationsWarning,
        stacklevel=4,
    )
    return restricted, dict(
        unimputed_observations=n,
        unimputed_treated_weight_share=share,
        unimputed_cells=cells,
    )

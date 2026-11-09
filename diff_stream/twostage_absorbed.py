"""Gardner's two-stage estimator with the fixed effects absorbed iteratively.

The first stage is BJS's: the untreated rows' model (unit, time and additional
fixed effects, covariates) fitted by hdfe-stream, whose predictions give every
row its first-stage residual. The second stage regresses those residuals on the
treatment indicators, one per event time or a single one. The indicators do not
overlap and there is no intercept, so a coefficient is the weighted mean of its
rows' residuals.

The covariance is Gardner's: each unit's second-stage score, net of the effect of
the estimated first stage on it. A row's weight in the first stage's effect on
coefficient m is -w_it z_it'u_m, with u_m = S^- Z'v_m the solution of the
untreated model's normal equations for the rows of m
(``HDFEResult.solve_normal_equations``), the computation that BJS uses per
cohort-period cell. Here there is one column per coefficient, so the solve is
small however many fixed-effect levels there are, and a single pass over the
residuals gives each unit's score.
"""

import warnings
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
from hdfe_stream.core.influence import ClusterScoreAccumulator

from ._core.partition import whole_group_batches
from .absorbed import (
    _DROPPED,
    absorbed_terms,
    first_columns,
    hdfe_options,
    info,
    solve_block,
    staged,
    untreated,
)


def _indicator(specification, horizons, anticipation):
    """The row's second-stage coefficient (an index), or null for none."""
    event = pl.col("t") - pl.col("g")
    if specification == "static":
        return (
            pl.when((pl.col("g") != 0) & (event >= -anticipation))
            .then(pl.lit(0, dtype=pl.Int64))
            .otherwise(None)
        )
    return (
        pl.when(pl.col("g") != 0)
        .then(
            event.replace_strict(
                horizons, list(range(len(horizons))), default=None, return_dtype=pl.Int64
            )
        )
        .otherwise(None)
    )


def fit_two_stage_absorbed(store, terms, specification, horizons, anticipation, *, threads=None):
    """(beta, covariance sums, diagnostics) of the two-stage fit, CRV0 by unit.

    ``horizons`` are the event times of a dynamic fit (empty for a static one).
    """
    from hdfe_stream import feols_stream

    frame, covariates, xs, absorbed = staged(store, terms)
    H = len(horizons) if specification == "dynamic" else 1
    fml = f"y ~ {' + '.join(xs) or '1'} | " + " + ".join(["u", "t", *absorbed])
    options = hdfe_options(store, threads, len(xs), vcov="iid", keep_intermediates=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", _DROPPED)
        first = feols_stream(
            fml, frame.filter(untreated(anticipation)), workdir=store.root, **options
        )
    try:
        with TemporaryDirectory(prefix="diff-stream-gardner-", dir=store.root) as workspace:
            beta, V = _second_stage(
                store, first, terms, frame, workspace, specification, horizons, anticipation, H
            )
        diagnostics = info(covariates, xs, list(first.coefnames), terms)
    finally:
        first.cleanup()
    diagnostics["first_stage_columns"] = diagnostics["regression_columns"]
    return beta, V, diagnostics


def _second_stage(store, first, terms, frame, workspace, specification, horizons, anticipation, H):
    weight = pl.col("w") if store.weighted else pl.lit(1.0)
    _, _, xs, absorbed = staged(store, terms)

    # Every row's first-stage residual: the outcome less the imputed untreated one.
    residuals = f"{workspace}/residuals.parquet"
    (
        first.predict(frame, name="prediction")
        .with_columns(pl.col("g").cast(pl.Int64), pl.col("t").cast(pl.Int64))
        .with_columns(
            (pl.col("y") - pl.col("prediction")).alias("r"),
            untreated(anticipation).alias("untreated"),
            _indicator(specification, horizons, anticipation).alias("m"),
        )
        .drop("prediction")
        .sink_parquet(residuals, row_group_size=store.read_rows, engine="streaming")
    )
    rows = pl.scan_parquet(residuals).filter(pl.col("m").is_not_null())
    unidentified = rows.select(pl.col("r").null_count()).collect().item()
    if unidentified:
        raise ValueError(
            f"Gardner's second stage is not identified for {unidentified:,} observations: "
            "a fixed-effect level of theirs has no untreated observation"
        )
    sums = (
        rows.group_by("m")
        .agg(
            (weight * pl.col("r")).sum().alias("effect"),
            (pl.col("w").sum() if store.weighted else pl.len().cast(pl.Float64)).alias("mass"),
        )
        .collect()
        .sort("m")
    )
    if sums.height != H or not (sums["mass"].to_numpy() > 0).all():
        raise ValueError("Gardner second-stage design is unidentified")
    mass = sums["mass"].to_numpy()
    beta = sums["effect"].to_numpy() / mass

    # The untreated model's normal equations for each coefficient's rows.
    names = [f"v{j}" for j in range(H)]
    target = rows.select(
        "u",
        "t",
        *[c for c in first_columns(first) if c not in ("u", "t")],
        *[
            pl.when(pl.col("m") == j).then(weight / mass[j]).otherwise(0.0).alias(n)
            for j, n in enumerate(names)
        ],
    )
    solution = first.solve_normal_equations(target, names, block=solve_block(store, H))
    try:
        V = _covariance(store, solution, residuals, names, xs, absorbed, beta, mass)
    finally:
        solution.cleanup()
    return beta, V


def _covariance(store, solution, residuals, names, xs, absorbed, beta, mass):
    """Sum over units of the outer products of their corrected second-stage scores."""
    H = len(names)
    tables = []
    for term in absorbed_terms(absorbed):
        table = solution.fixef(term).collect()
        keys = [c for c in table.columns if c not in names]
        tables.append((keys, table.with_row_index("__level"), table.select(names).to_numpy()))
    kept = solution.coef["coefficient"].to_list()
    coef = solution.coef.select(names).to_numpy()
    accumulator = ClusterScoreAccumulator()
    store._guard(32e6 + 8 * H * 4 * store.batch_units * store.periods)
    for d in whole_group_batches(residuals, "u", store.batch_units, store.read_rows):
        w = d["w"].to_numpy() if store.weighted else np.ones(d.height)
        r = d["r"].to_numpy()
        rowscore = np.zeros((d.height, H))
        # the first stage's part: a clean row's weight in each coefficient is
        # -w z'u, so its share of the score is that times its residual
        clean = np.flatnonzero(d["untreated"].to_numpy())
        if clean.size:
            c = d[clean]
            q = np.zeros((clean.size, H))
            for keys, table, values in tables:
                level = (
                    c.select(keys)
                    .with_columns(pl.col(k).cast(table.schema[k]) for k in keys)
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
                q += c.select(kept).to_numpy() @ coef
            rowscore[clean] = -(w[clean] * r[clean])[:, None] * q
        # the second stage's part: the rows of coefficient m, net of the mean
        m = d["m"].fill_null(-1).to_numpy()
        hit = np.flatnonzero(m >= 0)
        rowscore[hit, m[hit]] += w[hit] * (r[hit] - beta[m[hit]]) / mass[m[hit]]
        u = d["u"].to_numpy()
        starts = np.r_[0, np.flatnonzero(u[1:] != u[:-1]) + 1]
        accumulator.add_clusters(np.add.reduceat(rowscore, starts, axis=0))
    return accumulator.sums

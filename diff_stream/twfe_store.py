"""Observed-cell staging for the TWFE-weight diagnostic (no cohort restrictions)."""

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from ._core.design import CovariatePlan
from ._core.partition import GroupPartitions, Part, whole_group_batches
from ._core.weights import validate_weights
from .store import source


def inspect_weights(fml, data, panel, treatment, cohort, weights):
    if not isinstance(fml, str) or fml.count("~") != 1:
        raise ValueError("use an outcome formula such as 'y ~ x + z'")
    y, rhs = [s.strip() for s in fml.split("~")]
    if not y.isidentifier() or "|" in rhs:
        raise ValueError(
            "use a column-name outcome and covariates; unit/time effects are automatic"
        )
    if (treatment is None) == (cohort is None):
        raise ValueError("specify exactly one of treatment or cohort")
    d = treatment if treatment is not None else cohort
    names = [y, *panel, d] + ([] if weights is None else [weights])
    if len(panel) != 2 or any(not isinstance(n, str) or not n for n in names):
        raise ValueError(
            "panel must name unit/time columns; variable names must be nonempty strings"
        )
    if len(set(names)) != len(names):
        raise ValueError("outcome, unit, time, treatment/cohort and weights must be distinct")
    src = source(data)
    schema = src.collect_schema()
    if not schema[panel[1]].is_integer():
        raise ValueError("time must contain positive integers (gaps are permitted)")
    if not (schema[panel[0]].is_numeric() or schema[panel[0]] == pl.String):
        raise ValueError("unit IDs must be numeric or strings")
    for name in [y, d] + ([] if weights is None else [weights]):
        if not (schema[name].is_numeric() or (name == treatment and schema[name] == pl.Boolean)):
            raise ValueError(f"{name} must be numeric")
    if cohort is not None and not schema[cohort].is_integer():
        raise ValueError("cohort must contain integers, with 0 denoting never treated")
    plan = None if rhs in ("1", "0", "-1") else CovariatePlan.parse(y, rhs, schema)
    lf = src.select(
        pl.col(panel[0]).alias("id"),
        pl.col(panel[1]).cast(pl.Int64).alias("t"),
        pl.col(d)
        .cast(pl.Int64 if cohort is not None else pl.Float64)
        .alias("g" if cohort else "d"),
        pl.col(y).cast(pl.Float64).alias("y"),
        (pl.lit(1.0) if weights is None else pl.col(weights).cast(pl.Float64)).alias("w"),
        *(
            []
            if plan is None
            else [pl.col(c).alias(f"__raw{j}") for j, c in enumerate(plan.required)]
        ),
    )
    return lf, y, plan


class WeightStore(GroupPartitions):
    def __init__(self, lf, root, budget, preprocess_mb, batch_units, plan, panel_mode, varying):
        super().__init__(root, budget, preprocess_mb)
        self.plan, self.panel_mode, self.varying = plan, panel_mode, varying
        self.batch_units = batch_units
        self.paths = []
        self.n_units = self.n_obs = self.input_units = 0
        self.weight_scale = 0.0
        self.weights_time_invariant = True
        self._guard(preprocess_mb * 1e6 + 96e6)
        raw = self.root / "snapshot.parquet"
        lf.sink_parquet(raw, engine="streaming", row_group_size=self.read_rows)
        dates, size = set(), 0
        with pq.ParquetFile(raw, pre_buffer=False) as pf:
            self.input_rows = pf.metadata.num_rows
            for batch in pf.iter_batches(batch_size=self.read_rows):
                d = pl.from_arrow(batch)
                if sum(d.null_count().row(0)):
                    raise ValueError(
                        "missing values are unsupported; clean required columns explicitly"
                    )
                for c in ["y", "w", "d" if "d" in d.columns else "g"]:
                    if not d[c].is_finite().all():
                        raise ValueError("required numeric values must be finite")
                if d.schema["id"].is_numeric() and not d["id"].is_finite().all():
                    raise ValueError("unit IDs must be finite")
                if not (d["t"] > 0).all():
                    raise ValueError("time must contain positive integers")
                if "d" in d.columns and not d["d"].is_in([0.0, 1.0]).all():
                    raise ValueError("treatment must be binary (0 or 1)")
                if "g" in d.columns and not (d["g"] >= 0).all():
                    raise ValueError("cohort must be nonnegative, with 0 denoting never treated")
                validate_weights(d["w"].to_numpy())
                self.weight_scale = max(self.weight_scale, float(d["w"].max()))
                dates.update(d["t"].unique().to_list())
                self._guard(len(dates) * 128 + 32e6)
                size += batch.nbytes
                if plan is not None:
                    plan.discover(d)
        self.times = np.array(sorted(dates), dtype=np.int64)
        self.periods = len(dates)
        if self.periods < 2:
            raise ValueError("at least two observed periods are required")
        if plan is not None:
            plan.compile()
            self.extra_row_bytes = len(plan.names) * 8
        self._prepare(Part(raw, self.input_rows, size), 0)
        if not self.n_obs:
            raise ValueError("no observations remain after complete-panel selection")

    def _leaf(self, part):
        effective = part.size + part.rows * self.extra_row_bytes
        self._guard(effective * 4 + 32e6)
        with pq.ParquetFile(part.path, pre_buffer=False) as pf:
            d = pl.from_arrow(pf.read()).sort(["id", "t"])
        ids, t = d["id"].to_numpy(), d["t"].to_numpy()
        starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
        lengths = np.diff(np.r_[starts, len(d)])
        u = np.repeat(np.arange(len(starts)), lengths)
        if np.any((ids[1:] == ids[:-1]) & (t[1:] == t[:-1])):
            raise ValueError("duplicate unit-time keys are unsupported")
        w = d["w"].to_numpy()
        constant = np.all(w == w[starts][u])
        if not constant and not self.varying:
            raise ValueError("time-varying weights require allow_time_varying_weights=True")
        self.weights_time_invariant &= bool(constant)
        if "g" in d.columns:
            g = d["g"].to_numpy()
            if np.any(g != g[starts][u]):
                raise ValueError("cohort must be constant within unit")
            d = d.with_columns(
                ((pl.col("g") > 0) & (pl.col("t") >= pl.col("g"))).cast(pl.Float64).alias("d")
            )
        # Evaluate source covariates before sample deletion, as in PanelStore.
        if self.plan is not None:
            d = d.hstack(self.plan.evaluate(d))
        self.input_units += len(starts)
        complete = lengths == self.periods
        if self.panel_mode == "balanced" and not complete.all():
            raise ValueError("balanced panels required; set incomplete_units='keep' or 'drop'")
        if self.panel_mode == "complete":
            d = d.filter(pl.Series(complete[u]))
            lengths = lengths[complete]
        if len(d):
            d = d.select("id", "t", "y", "d", "w", *[c for c in d.columns if c.startswith("__x")])
            path = self._name("units")
            d.write_parquet(path, row_group_size=self.read_rows)
            self.paths.append(path)
            self.n_units += len(lengths)
            self.n_obs += len(d)
            self.partition_count += 1
            self.max_leaf_bytes = max(self.max_leaf_bytes, effective)
        part.path.unlink()

    def batches(self):
        for path in self.paths:
            yield from whole_group_batches(path, "id", self.batch_units, self.read_rows)

    def matrices(self):
        cols = (
            []
            if self.plan is None
            else [j for j, n in enumerate(self.plan.names) if n != "Intercept"]
        )
        T, P = self.periods, self.periods - 1 + len(cols)
        for frame in self.batches():
            ids = frame["id"].to_numpy()
            starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
            lengths = np.diff(np.r_[starts, len(frame)])
            u = np.repeat(np.arange(len(starts)), lengths)
            t = np.searchsorted(self.times, frame["t"].to_numpy())
            n = len(starts)
            X = np.zeros((n, T, P))
            time = t > 0
            X[u[time], t[time], t[time] - 1] = 1
            for k, j in enumerate(cols):
                X[u, t, T - 1 + k] = frame[f"__x{j}"].to_numpy()
            w, D, Y = np.zeros((n, T)), np.zeros((n, T)), np.zeros((n, T))
            w[u, t] = frame["w"].to_numpy() / self.weight_scale
            validate_weights(w[u, t])
            D[u, t], Y[u, t] = frame["d"].to_numpy(), frame["y"].to_numpy()
            yield frame, X, D, Y, w, u, t

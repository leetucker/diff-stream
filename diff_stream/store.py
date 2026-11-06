"""Snapshot, recursively partition whole units, validate and sort bounded leaves."""

from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from ._core.partition import GroupPartitions, Part
from ._core.weights import validate_weights


def source(data):
    if isinstance(data, pl.DataFrame):
        return data.lazy()
    if isinstance(data, pl.LazyFrame):
        return data
    if isinstance(data, (str, Path)):
        return pl.scan_parquet(str(data), low_memory=True)
    raise TypeError("data must be a Polars frame or a Parquet path/glob")


def inspect(fml, data, panel, cohort, *, allow_covariates=False, weights=None, allow_fe=False):
    """Compile the projection and check its schema without scanning rows."""
    if not isinstance(fml, str) or fml.count("~") != 1:
        raise ValueError("use a single outcome formula: 'y ~ 1'")
    y, rhs = [s.strip() for s in fml.split("~")]
    if "|" in rhs and allow_fe:
        from .linear import regression_formula

        rhs = regression_formula(rhs, panel)
    if not y.isidentifier() or "|" in rhs or (rhs != "1" and not allow_covariates):
        raise NotImplementedError("the first milestone supports column-name outcomes and '~ 1'")
    if len(panel) != 2 or len(set([y, *panel, cohort])) != 4:
        raise ValueError("outcome, unit, time and cohort must be distinct column names")
    unit, time = panel
    src = source(data)
    if weights is not None:
        if not isinstance(weights, str) or not weights:
            raise ValueError("weights must be a sampling-weight column name or None")
        if weights in (y, unit, time, cohort):
            raise ValueError(
                "sampling weights must be distinct from outcome, unit, time and cohort"
            )
        ws = src.collect_schema()[weights]
        if not ws.is_numeric() or ws == pl.Boolean:
            raise ValueError("sampling weights must be numeric")
    plan = None
    if rhs != "1":
        from ._core.design import CovariatePlan

        plan = CovariatePlan.parse(y, rhs, src.collect_schema())
    lf = src.select(
        pl.col(unit).alias("id"),
        pl.col(time).alias("t"),
        pl.col(cohort).alias("g"),
        pl.col(y).cast(pl.Float64).alias("y"),
        *([] if weights is None else [pl.col(weights).cast(pl.Float64).alias("w")]),
        *(
            []
            if plan is None
            else [pl.col(c).alias(f"__raw{j}") for j, c in enumerate(plan.required)]
        ),
    )
    schema = lf.collect_schema()
    if not schema["t"].is_integer() or not schema["g"].is_integer():
        raise ValueError("time and cohort must be integers; 0 denotes never treated")
    if not (schema["id"].is_integer() or schema["id"].is_float() or schema["id"] == pl.String):
        raise ValueError("unit IDs must be numeric or strings")
    return lf.with_columns(pl.col("t").cast(pl.Int64), pl.col("g").cast(pl.Int64)), y, plan


class PanelStore(GroupPartitions):
    """Every leaf contains whole units. No table of all unit IDs is retained.

    preprocess_mb budgets decoded partition data plus sort/validation copies;
    readers/writers and the source LazyFrame have separate runtime overhead.
    """

    def __init__(
        self,
        lf,
        root,
        batch_units,
        budget,
        preprocess_mb=64,
        design=None,
        panel_mode="balanced",
        anticipation=0,
        *,
        require_never_treated=True,
        retain_actual_cohort=False,
        allow_time_varying_weights=False,
        allow_always_treated=False,
        actual_cohort_column=None,
    ):
        super().__init__(root, budget, preprocess_mb)
        self.batch_units = batch_units
        self.panel_mode = panel_mode
        self.anticipation = anticipation
        self.retain_actual_cohort = retain_actual_cohort
        self.actual_cohort_column = actual_cohort_column
        if actual_cohort_column is not None and (
            not retain_actual_cohort
            or actual_cohort_column not in lf.collect_schema()
            or not lf.collect_schema()[actual_cohort_column].is_integer()
        ):
            raise ValueError("an original cohort column requires retained integer adoption dates")
        self.allow_time_varying_weights = allow_time_varying_weights
        self.allow_always_treated = allow_always_treated
        if allow_always_treated and anticipation:
            raise ValueError("always-treated storage is only available without anticipation")
        self.weights_time_invariant = True
        self.actual_gc = {}
        self.actual_cell_counts = {}
        self.actual_cell_weights = {}
        self.is_balanced = True
        self.n_units_input = 0
        self.n_rows_used = 0
        self.cell_counts = {}
        self.cell_weights = {}
        self.gw = {}
        self.gw_observed_mean = {}
        self.weighted = "w" in lf.collect_schema()
        self.weight_scale = 1.0 if not self.weighted else 0.0
        self.design = design
        self.x_time_invariant = True
        self.paths = []
        self.gc = {}
        self.sums = {}
        self.n_units = 0
        self._guard(preprocess_mb * 1e6 + 96e6)
        raw = self.root / "snapshot.parquet"
        lf.sink_parquet(raw, engine="streaming", row_group_size=self.read_rows)
        times = set()
        size = 0
        with pq.ParquetFile(raw, pre_buffer=False) as pf:
            self.n_rows = pf.metadata.num_rows
            for b in pf.iter_batches(batch_size=self.read_rows):
                self._guard(16e6)
                d = pl.from_arrow(b)
                if sum(d.null_count().row(0)) or not d["y"].is_finite().all():
                    raise ValueError(
                        "missing or non-finite values are not supported; clean the panel explicitly"
                    )
                if d.schema["id"].is_float() and not d["id"].is_finite().all():
                    raise ValueError("unit IDs must be finite")
                if self.weighted:
                    w = d["w"].to_numpy()
                    validate_weights(w)
                    self.weight_scale = max(self.weight_scale, float(w.max()))
                times.update(d["t"].unique().to_list())
                # Prevent an invalid long time grid from growing unchecked.
                self._guard(len(times) * 128 + 16e6)
                size += b.nbytes
                if self.design is not None:
                    self.design.discover(d)
        self.times = np.array(sorted(times), dtype=np.int64)
        self.periods = len(times)
        if self.periods < 2 or self.times[0] <= 0 or np.any(np.diff(self.times) != 1):
            raise ValueError("at least two consecutive positive integer periods are required")
        if self.design is not None:
            self.design.compile()
            self.extra_row_bytes = len(self.design.names) * 8
        if self.retain_actual_cohort:
            self.extra_row_bytes += 8
        self._prepare(Part(raw, self.n_rows, size), 0)
        self.unit_weight_anchor = "first observed period per retained unit"
        if not self.is_balanced and not self.weights_time_invariant:
            self.gw = self.gw_observed_mean
            self.unit_weight_anchor = "mean observed weight per retained unit"
        if require_never_treated and (0 not in self.gc or len(self.gc) < 2):
            raise ValueError("at least one never-treated and one treated cohort are required")
        if not require_never_treated and len(self.gc) < 2:
            raise ValueError(
                "at least two distinct treatment-timing groups are required; an all-never-treated sample has no treatment effects"
            )
        self.weight_normalizer = self.n_units / sum(self.gw.values())
        self.gw = {g: v * self.weight_normalizer for g, v in self.gw.items()}
        self.cell_weights = {g: v * self.weight_normalizer for g, v in self.cell_weights.items()}
        self.actual_cell_weights = {
            g: v * self.weight_normalizer for g, v in self.actual_cell_weights.items()
        }
        self.means = np.vstack(
            [
                np.divide(
                    self.sums[g] * self.weight_normalizer,
                    self.cell_weights[g],
                    out=np.full(self.periods, np.nan),
                    where=self.cell_counts[g] > 0,
                )
                for g in sorted(self.gc)
            ]
        )
        del self.sums

    def _oversized_group(self, part):
        if self.panel_mode == "balanced" and part.rows != self.periods:
            raise ValueError("balanced panels with unique unit-time keys are required")
        return super()._oversized_group(part)

    def _leaf(self, part):
        effective = part.size + part.rows * self.extra_row_bytes
        self._guard(effective * 4 + 32e6)
        with pq.ParquetFile(part.path, pre_buffer=False) as pf:
            d = pl.from_arrow(pf.read()).sort(["id", "t"])
        self.max_leaf_bytes = max(self.max_leaf_bytes, effective)
        ids = d["id"].to_numpy()
        starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
        lengths = np.diff(np.r_[starts, len(ids)])
        t = d["t"].to_numpy()
        g = d["g"].to_numpy()
        unit = np.repeat(np.arange(len(starts)), lengths)
        if np.any((ids[1:] == ids[:-1]) & (t[1:] == t[:-1])):
            raise ValueError(
                "balanced panels with unique unit-time keys are required; duplicate unit-time key"
            )
        if np.any(g != g[starts][unit]):
            raise ValueError("cohort must be constant within unit")
        if self.weighted:
            w = d["w"].to_numpy()
            if np.any(w != w[starts][unit]) and not self.allow_time_varying_weights:
                raise ValueError(
                    "sampling weights must be constant within unit; time-varying weights are unsupported"
                )
        cohort = g[starts].copy()
        actual = cohort.copy() if self.retain_actual_cohort else None
        if self.actual_cohort_column is not None:
            original = d[self.actual_cohort_column].to_numpy()
            if np.any(original < 0) or np.any(original != original[starts][unit]):
                raise ValueError("original cohort must be nonnegative and constant within unit")
            actual = original[starts].copy()
        if np.any(cohort < 0):
            raise ValueError(
                "always-treated/negative cohorts are unsupported; exclude them explicitly"
            )
        onset = cohort - self.anticipation
        if self.allow_always_treated:
            cohort[(cohort > 0) & (onset <= self.times[0])] = self.times[0]
        elif np.any((cohort != 0) & (onset <= self.times[0])):
            raise ValueError(
                "always-treated/negative cohorts or anticipation without a clean pre-period are unsupported; exclude them explicitly"
            )
        cohort[onset > self.times[-1]] = 0
        self.n_units_input += len(starts)
        complete = lengths == self.periods
        if self.panel_mode == "balanced" and not complete.all():
            raise ValueError(
                "balanced panels with unique unit-time keys are required; select panel_mode='unbalanced' or 'complete'"
            )
        if self.panel_mode == "complete":
            keep = complete[unit]
            d = d.filter(pl.Series(keep))
            t = t[keep]
            cohort = cohort[complete]
            if actual is not None:
                actual = actual[complete]
            lengths = lengths[complete]
            unit = np.repeat(np.arange(len(cohort)), lengths)
        else:
            self.is_balanced &= bool(complete.all())
        if not len(cohort):
            part.path.unlink()
            return
        y = d["y"].to_numpy()
        w = d["w"].to_numpy() / self.weight_scale if self.weighted else np.ones(len(y))
        validate_weights(w)
        first = np.r_[0, np.cumsum(lengths)[:-1]]
        self.weights_time_invariant &= bool(np.all(w == w[first][unit]))
        mean_weight = np.bincount(unit, weights=w, minlength=len(cohort)) / lengths
        ti = t - self.times[0]
        for k in np.unique(cohort):
            selected = cohort == k
            rows = selected[unit]
            self.gc[int(k)] = self.gc.get(int(k), 0) + int(selected.sum())
            self.gw[int(k)] = self.gw.get(int(k), 0.0) + float(w[first][selected].sum())
            self.gw_observed_mean[int(k)] = self.gw_observed_mean.get(int(k), 0.0) + float(
                mean_weight[selected].sum()
            )
            sums = np.bincount(ti[rows], weights=y[rows] * w[rows], minlength=self.periods)
            mass = np.bincount(ti[rows], weights=w[rows], minlength=self.periods)
            self.cell_weights[int(k)] = self.cell_weights.get(int(k), np.zeros(self.periods)) + mass
            counts = np.bincount(ti[rows], minlength=self.periods)
            self.sums[int(k)] = self.sums.get(int(k), np.zeros(self.periods)) + sums
            self.cell_counts[int(k)] = (
                self.cell_counts.get(int(k), np.zeros(self.periods, dtype=int)) + counts
            )
        if actual is not None:
            for k in np.unique(actual):
                selected = actual == k
                rows = selected[unit]
                self.actual_gc[int(k)] = self.actual_gc.get(int(k), 0) + int(selected.sum())
                self.actual_cell_counts[int(k)] = self.actual_cell_counts.get(
                    int(k), np.zeros(self.periods, dtype=int)
                ) + np.bincount(ti[rows], minlength=self.periods)
                self.actual_cell_weights[int(k)] = self.actual_cell_weights.get(
                    int(k), np.zeros(self.periods)
                ) + np.bincount(ti[rows], weights=w[rows], minlength=self.periods)
        self._guard(32e6 + 32 * (len(self.gc) + len(self.actual_gc)) * self.periods)
        path = self._name("units")
        output = pl.DataFrame({"u": unit + self.n_units, "t": t, "g": cohort[unit], "y": y})
        if actual is not None:
            output = output.with_columns(pl.Series("g_actual", actual[unit]))
        if self.weighted:
            output = output.with_columns(pl.Series("w", w))
        self.n_units += len(cohort)
        self.n_rows_used += len(y)
        if self.design is not None:
            x = self.design.evaluate(d)
            self._guard(x.estimated_size() * 3 + 32e6)
            output = output.hstack(x)
            xm = x.to_numpy()
            first = np.r_[0, np.cumsum(lengths)[:-1]]
            self.x_time_invariant &= bool(np.all(xm == xm[first][unit]))
        output.write_parquet(path, row_group_size=self.read_rows)
        self.paths.append(path)
        self.partition_count += 1
        part.path.unlink()

    def batches(
        self, *, include_x=False, include_weights=False, actual_cohort=False, row_weights=False
    ):
        if actual_cohort and not self.retain_actual_cohort:
            raise ValueError("original cohorts were not retained for this fit")
        if not self.is_balanced:
            yield from self._sparse_batches(
                include_x=include_x,
                include_weights=include_weights,
                actual_cohort=actual_cohort,
                row_weights=row_weights,
            )
            return
        for path in self.paths:
            with pq.ParquetFile(path, pre_buffer=False) as pf:
                K = len(self.design.names) if self.design is not None else 1
                columns = ["g_actual" if actual_cohort else "g", "y"] + (
                    [f"__x{j}" for j in range(K)] if include_x and self.design is not None else []
                )
                if include_weights and self.weighted:
                    columns.append("w")
                for b in pf.iter_batches(
                    batch_size=self.batch_units * self.periods, columns=columns
                ):
                    g = b.column(0).to_numpy().reshape(-1, self.periods)[:, 0].copy()
                    y = b.column(1).to_numpy().reshape(-1, self.periods).copy()
                    result = (g, y)
                    if include_x and self.design is not None:
                        x = np.column_stack([b.column(j + 2).to_numpy() for j in range(K)])
                        result += (x.reshape(-1, self.periods, K),)
                    elif include_x:
                        result += (np.ones((*y.shape, 1)),)
                    if include_weights:
                        if row_weights:
                            w = (
                                b.column("w").to_numpy().reshape(-1, self.periods)
                                * self.weight_normalizer
                                if self.weighted
                                else np.ones(y.shape)
                            )
                        else:
                            w = (
                                b.column("w").to_numpy().reshape(-1, self.periods)[:, 0]
                                * self.weight_normalizer
                                if self.weighted
                                else np.ones(len(g))
                            )
                        result += (w,)
                    yield result

    def _sparse_batches(
        self, *, include_x=False, include_weights=False, actual_cohort=False, row_weights=False
    ):
        """Pad only a numerical batch, never a whole sparse partition."""
        from ._core.partition import whole_group_batches

        K = len(self.design.names) if self.design is not None else 1
        for path in self.paths:
            for d in whole_group_batches(path, "u", self.batch_units, self.read_rows):
                u = d["u"].to_numpy()
                starts = np.r_[0, np.flatnonzero(u[1:] != u[:-1]) + 1]
                lengths = np.diff(np.r_[starts, len(u)])
                row = np.repeat(np.arange(len(starts)), lengths)
                ti = d["t"].to_numpy() - self.times[0]
                self._guard(32e6 + 8 * len(starts) * self.periods * (K + 1))
                y = np.full((len(starts), self.periods), np.nan)
                y[row, ti] = d["y"].to_numpy()
                g = d["g_actual" if actual_cohort else "g"].to_numpy()[starts].copy()
                result = (g, y)
                if include_x:
                    x = np.full((len(starts), self.periods, K), np.nan)
                    x[row, ti] = (
                        d.select([f"__x{j}" for j in range(K)]).to_numpy()
                        if self.design is not None
                        else 1.0
                    )
                    result += (x,)
                if include_weights:
                    if row_weights:
                        w = np.zeros(y.shape)
                        w[row, ti] = (
                            d["w"].to_numpy() * self.weight_normalizer if self.weighted else 1.0
                        )
                    else:
                        w = (
                            d["w"].to_numpy()[starts] * self.weight_normalizer
                            if self.weighted
                            else np.ones(len(g))
                        )
                    result += (w,)
                yield result

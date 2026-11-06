"""CS-specific comparison selection around shared weighted balance kernels.

Optional one-comparison-at-a-time staging avoids retaining all unit covariates
or propensity scores on a result. Regression adjustment does not reweight controls.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from hdfe_stream.core.balance import effective_sizes, weighted_balance

from ._core.contrasts import att_parts


class CSBalanceCollector:
    def __init__(self, store, ecdf):
        self.store, self.ecdf = store, ecdf
        names = store.design.names if store.design is not None else []
        self.keep = [j for j, name in enumerate(names) if name != "Intercept"]
        if not self.keep:
            raise ValueError("balance=True requires non-intercept CS covariates")
        self.names = [names[j] for j in self.keep]
        self.columns = [f"__x{j}" for j in range(len(self.keep))]
        self.tables = []
        self.source_passes = 0

    def compute(self, rows, adjusted):
        """Shared kernels consume a lazy one-comparison temporary Parquet file."""
        with TemporaryDirectory(prefix="cs-balance-", dir=self.store.root) as run:
            path = Path(run) / "rows.parquet"
            writer = None
            try:
                for X, d, w, after in rows():
                    if not len(d):
                        continue
                    self.store._guard(32e6 + X.nbytes * 4)
                    frame = pl.DataFrame(
                        {
                            **{
                                name: X[:, k]
                                for name, k in zip(self.columns, self.keep, strict=True)
                            },
                            "__d": d,
                            "__before": w,
                            "__after": after,
                        }
                    )
                    table = frame.to_arrow()
                    if writer is None:
                        writer = pq.ParquetWriter(path, table.schema)
                    writer.write_table(table, row_group_size=self.store.read_rows)
            finally:
                if writer is not None:
                    writer.close()
            self.source_passes += 1
            if writer is None:
                raise ValueError("CS balance comparison has no observations")
            lf = pl.scan_parquet(path, low_memory=True)
            before = weighted_balance(lf, "__d", self.columns, "__before", ecdf=self.ecdf)
            scales = {
                row["term"]: float(np.sqrt((row["var1"] + row["var0"]) / 2))
                for row in before.iter_rows(named=True)
            }
            tables = []
            for stage, table, weight in [
                ("before", before, "__before"),
                *([("after", None, "__after")] if adjusted else []),
            ]:
                if table is None:
                    table = weighted_balance(
                        lf, "__d", self.columns, weight, scale=scales, ecdf=self.ecdf
                    )
                sizes = effective_sizes(lf, "__d", weight)
                tables.append(
                    table.with_columns(
                        pl.col("term").replace_strict(
                            dict(zip(self.columns, self.names, strict=True))
                        ),
                        pl.lit(stage).alias("stage"),
                        pl.lit(sizes.n1, dtype=pl.Int64).alias("n_treated"),
                        pl.lit(sizes.n0, dtype=pl.Int64).alias("n_controls"),
                        pl.lit(sizes.ess1).alias("ess_treated"),
                        pl.lit(sizes.ess0).alias("ess_controls"),
                    )
                )
            return pl.concat(tables)

    def balanced(self, comparisons, fitted, cells, batches):
        for c, f in zip(comparisons, fitted, strict=True):

            def rows(c=c, f=f):
                for raw in batches():
                    mask, X, d, _ = c.select(raw)
                    sw = c.weights(raw, mask)
                    wc = (
                        att_parts(X, d, f.nuisance.beta, f.control_max)[1]
                        if f.method != "reg"
                        else np.ones(len(d))
                    )
                    yield X, d, sw, sw * np.where(d, 1, wc)

            table = self.compute(rows, f.method != "reg")
            for k, j in enumerate(c.cells):
                self.tables.append(
                    table.with_columns(
                        pl.lit(c.cohort, dtype=pl.Int64).alias("cohort"),
                        pl.lit(cells[j][1], dtype=pl.Int64).alias("time"),
                        pl.lit(int(self.store.times[c.bases[k]]), dtype=pl.Int64).alias("base"),
                        pl.lit("comparison").alias("stratum"),
                        pl.lit(int(self.store.times[c.xperiod]), dtype=pl.Int64).alias(
                            "covariate_time"
                        ),
                    )
                )

    def unbalanced(self, comparisons, fitted, batches):
        for c, f in zip(comparisons, fitted, strict=True):
            for use_post, stratum in [(False, "pre"), (True, "post")]:

                def rows(c=c, f=f, use_post=use_post):
                    for raw in batches():
                        unit, X, d, _, post = c.select(raw)
                        sw = c.weights(raw, unit, post)
                        wc = (
                            att_parts(X, d, f.system.beta, f.trim)[1]
                            if f.method != "reg"
                            else np.ones(len(d))
                        )
                        mask = post == use_post
                        yield X[mask], d[mask], sw[mask], (sw * np.where(d, 1, wc))[mask]

                table = self.compute(rows, f.method != "reg")
                self.tables.append(
                    table.with_columns(
                        pl.lit(c.cohort, dtype=pl.Int64).alias("cohort"),
                        pl.lit(int(self.store.times[c.target]), dtype=pl.Int64).alias("time"),
                        pl.lit(int(self.store.times[c.base]), dtype=pl.Int64).alias("base"),
                        pl.lit(stratum).alias("stratum"),
                        pl.lit(int(self.store.times[c.target if use_post else c.base])).alias(
                            "covariate_time"
                        ),
                    )
                )

    def table(self):
        return pl.concat(self.tables).sort("cohort", "time", "stratum", "stage", "term")

# Quickstart

## The data

A panel in long form: one row per unit and period, with an outcome, the
unit and period identifiers, and the period each unit is first treated (0
or null for units never treated); a 0/1 treatment column works too
(`treatment=` instead of `cohort=`). Treatment is absorbing: once treated, a
unit stays treated. Periods can be every year, every second year, and so on. The data can be a Polars DataFrame or LazyFrame, or Parquet files.

| person | year | first_treated_year | earnings |
|---|---|---|---|
| 1 | 2015 | 2018 | 41,200 |
| 1 | 2016 | 2018 | 42,050 |
| … | … | … | … |
| 2 | 2015 | 0 | 38,900 |

`diff_stream.simulate.simulate_panel` draws such a panel with effects that vary
by cohort and by time since treatment:

```python
from diff_stream.simulate import simulate_panel

data = simulate_panel(2000, 8)   # 2,000 units, periods 1-8
data.head()
```

## Fit and aggregate

Every estimator takes the same first arguments: a formula, the data, the unit
and time columns, and the cohort column.

```python
from diff_stream import did_callaway_santanna

fit = did_callaway_santanna("y ~ 1", data, panel=("id", "time"), cohort="first_treat")
```

The fit holds one effect per cohort and period (`fit.cohort_time()` lists
them, with their sample sizes). Aggregate them for the summaries you report:

```python
fit.aggregate().summary()            # event study: effects by time since treatment
fit.aggregate("overall").summary()   # one ATT over all treated cells
fit.aggregate("cohort").tidy()       # by cohort
fit.aggregate("calendar").tidy()     # by calendar period
```

Each aggregate is a result of its own, with the methods of hdfe-stream's
results: `tidy()` (a Polars DataFrame of estimates, standard errors and
intervals), `confint()`, `wald_test()`, `summary()`, and `etable()` for a table
of several models. Standard errors are clustered by unit.

## Another estimator

The other estimators take the same arguments, so comparing them is a loop:

```python
from diff_stream import did_bjs, did_sun_abraham

for estimator in [did_callaway_santanna, did_sun_abraham, did_bjs]:
    fit = estimator("y ~ 1", data, panel=("id", "time"), cohort="first_treat")
    print(estimator.__name__, fit.aggregate("overall").tidy())
```

See [Choosing an estimator](choosing.md) for how they differ, and the
[examples](examples/index.md) for covariates, weights, unbalanced panels,
pre-trend tests and the bootstrap.

## Larger data

Pass a Parquet path or glob instead of a DataFrame; the data are read in
batches and need not fit in memory. On a cluster, limit the memory and threads
a fit may use (they default to what the job was allocated):

```python
fit = did_callaway_santanna(
    "earnings ~ age + C(region)",
    "panel/*.parquet",
    panel=("person", "year"),
    cohort="first_treated_year",
    n_threads=8,
    max_memory_gb=32,
)
```

See [Memory and threads](resources.md).

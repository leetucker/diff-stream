# Results and aggregation

The four estimators that estimate one effect per cohort and period return a
`DIDResult` (the two-stage estimator returns its coefficients directly; see
[Two-stage DiD](estimators/two_stage.md)). A `DIDResult` has hdfe-stream's
result methods:

| Method | |
|---|---|
| `tidy()` | a Polars DataFrame: estimate, standard error, t and p values, interval |
| `summary()`, `summary_dict()`, `summary_json()` | printed, dict and JSON summaries |
| `confint(alpha, joint=False)` | intervals; `joint=True` gives simultaneous ones from the covariance |
| `wald_test(R, q)` | a joint Wald test |
| `with_vcov("CRV1")` | the same fit with the other clustered covariance |
| `etable(...)` | a table of several fits (`diff_stream.etable`, through pyfixest) |

## The cohort-period effects

`fit.cohort_time()` lists one row per estimated effect: its `cohort`, `time`,
`event_time` (time minus cohort) and `period_type` (pre-treatment,
anticipation or post-treatment), with the estimate and the units behind it
(`weight_units`, `observed_units`, `cohort_units`, and `weight_total` with
sampling weights). Coefficient names are `g::<cohort>:t::<time>`.

## Aggregations

`fit.aggregate(kind)` averages the cohort-period effects. Each returns a
`DIDResult` with the joint covariance of its estimates.

| `kind` | Reports | Averages |
|---|---|---|
| `"event_time"` (default) | `event::<e>` for each time since treatment | the cohorts observed at that event time |
| `"overall"` | `ATT` | all post-treatment cells |
| `"cohort"` | `cohort::<g>` | each cohort's post-treatment periods |
| `"calendar"` | `calendar::<t>` | the cohorts treated in each period |

Cells are weighted by their units (sampling-weight totals with weights).
Callaway-Sant'Anna weights by each cohort's units, as R's did does;
Sun-Abraham, BJS and ETWFE weight by each cell's observed units, as fixest,
didimputation and emfx do. The aggregate standard errors of Callaway-Sant'Anna
and ETWFE (unless `treat_sample_as_fixed=True`) include the uncertainty in
those estimated shares. BJS recomputes its standard errors for every
aggregate.

```python
fit.aggregate("cohort", horizons=[0, 1, 2])   # the first three periods only
fit.aggregate(balanced_horizon=2)             # cohorts observed through e = 2
fit.aggregate(balance_window=(-3, 2))         # cohorts observed at every e in -3..2
```

- **`horizons`** keeps only the listed event times, for every kind.
- **`balanced_horizon=E`** keeps only the cohorts with effects at every event
  time from 0 to E, so the event study's composition does not change over the
  post-treatment periods (R did's `balance_e`).
- **`balance_window=(lo, hi)`** keeps only the cohorts with effects at every
  event time in the window, and reports that window.

These balance cohorts, not units: in an unbalanced panel the units observed
can still change from period to period.

## What an aggregate averages

```python
event_study = fit.aggregate()
event_study.contributions()   # one row per cell: its weight and estimate
event_study.composition()     # per estimate: cohorts, their weights and units
fit.contributions("calendar") # inspect an aggregation before computing it
```

`composition()` flags (`cohorts_changed`, `cohort_weights_changed`) where the
cohorts behind an event study change from one event time to the next, a
common source of misleading event-study shapes.
`diagnostics["excluded_cohorts"]` lists the cohorts a balancing option dropped
and why.

## Pre-trend tests

`fit.pretrend_test()` is a Wald test that the clean pre-treatment effects are
jointly zero. It returns a dictionary with the statistic, degrees of freedom,
p value and exactly which coefficients, cells and event times were tested.

- On a Callaway-Sant'Anna or Sun-Abraham fit, it tests the pre-treatment
  cohort-period effects; `level="event_time"` tests their event-time averages
  instead, and `horizons=[-4, -3]` selects event times.
- On an event-study aggregate, it tests its pre-treatment event times.
- BJS needs `pretrend_leads=L` at fit time (see [BJS](estimators/bjs.md)).

Anticipation periods and the omitted reference are never tested.

## Temporary files

Weighted or unbalanced BJS results keep temporary files for further
aggregation. `fit.cleanup()` deletes them (aggregates already computed still
work), as does using the result as a context manager:

```python
with did_bjs("y ~ 1", data, panel=("id", "time"), cohort="first_treat",
             weights="w") as fit:
    fit.aggregate().summary()
```

# Aggregation and sample composition

All three estimators support these aggregations of their original group-time
estimates:

| `kind` | Included cells | Weights within each reported estimate |
|---|---|---|
| `"dynamic"` (default) | Same event time `time - cohort` | Cohort/cell unit counts |
| `"simple"` | All selected post-treatment cells | Cohort/cell unit counts |
| `"cohort"` (alias `"group"`) | Post-treatment cells within each cohort | Cohort/cell unit counts |
| `"calendar"` | Post-treatment cohorts observed in each calendar period | Cohort/cell unit counts |

CS uses each cohort's unique retained-unit count, as in R `did::aggte`. Its
within-cohort aggregate therefore averages selected periods equally. SA and
BJS use each cell's observed-unit count; an unbalanced cohort's periods can
receive different weights. CS includes estimated-cohort-share uncertainty in
joint covariance. SA conditions on treatment timing/observed support, and BJS
recomputes smart residual averages and optional leave-out adjustments for every
requested aggregation. [R's aggregation reference](https://bcallaway11.github.io/did/reference/aggte.html)
describes its group, calendar and exposure-weighting conventions.

```python
from diff_stream import cs_stream

fit = cs_stream("y ~ 1", data, panel=("id", "time"), cohort="first_treat")
by_cohort = fit.aggregate("cohort")
by_calendar = fit.aggregate("calendar")
short_run = fit.aggregate("cohort", horizons=[0, 1, 2])
```

Each cohort/calendar call returns a vector, with joint covariance and the usual
result methods. `horizons` selects event times for every aggregation kind.
Non-dynamic kinds always exclude pre-treatment cells. In particular, calendar
aggregation applies `horizons` if supplied; R ignores its analogous event-time
limits for calendar aggregation. The default unrestricted calendar results
match the R reference.

## Inspecting contributors

```python
event_study = fit.aggregate()
print(event_study.contributions())
print(event_study.composition())

# Inspect a proposed aggregation without computing another covariance:
print(fit.contributions("calendar"))
print(fit.composition("dynamic", horizons=[0, 1, 2]))
```

`contributions()` reports one row per nonzero cell weight:

| Column | Meaning |
|---|---|
| `aggregate`, `cell` | Reported estimate and source group-time coefficient |
| `cohort`, `time`, `event_time` | Source-cell coordinates |
| `estimate`, `weight`, `weighted_estimate` | Cell estimate, aggregation weight and their product |
| `weight_units` | Count used to form the weight: cohort units for CS, observed cell units for SA/BJS |
| `observed_units` | Units with an observed outcome in this treated cohort at the target calendar date |
| `cohort_units` | Unique units in that cohort anywhere in the retained estimation sample |

Weights sum to one within each reported estimate; `weighted_estimate` sums to
its point estimate. Observed counts are target-cell counts, not the full
baseline/target comparison sample or control sample. For CS, consult fit
comparison diagnostics for nuisance-fit and propensity-trimming populations.
An observed-date count can be smaller than the cohort count in an unbalanced
panel. `group_time()` now exposes these three count columns too; its existing
`n_units` column remains an alias for `weight_units`.

`composition()` summarizes each reported estimate with:

- `n_cohorts`, `n_cells`, and the contributing `cohorts` list;
- `cohort_weights`, aligned with the cohort list;
- `cohort_units`, counting each contributing cohort's population once;
- `observed_unit_periods`, summing observed target-cell rows;
- `cohorts_changed` and `cohort_weights_changed`, comparing with the preceding
  reported row (both false for the first row).

`observed_unit_periods` can count a unit more than once when several dates are
averaged. It is not a count of distinct people across dates. Changing contributor
sets or weights is especially relevant for event-study curves: the population
represented by the estimate can change across horizons.

Diagnostics retain the weighting convention, contributing/excluded cohorts,
exclusion reasons, balancing settings, and composition-change flags. A dynamic
summary flags actual changes in cohort membership or weights. Aggregating does
not refit the estimator, change its declared sample, or alter the original
result's diagnostics. Its `n_obs` remains the full fitted sample size.

## Balancing event-time support

For R-style post-treatment exposure balancing:

```python
balanced_post = fit.aggregate(balance_e=2)
print(balanced_post.composition())
```

`balance_e=E` retains cohorts with supported cells at every event time `0..E`.
As in R, reported event times lie between `E - (Tmax - Tmin)` and `E`, subject
to available estimates and any `horizons` filter. The support check happens
before that display filter. `E` must be a nonnegative integer. This holds the
cohort set fixed across the post-treatment events `0..E`; pre-treatment
contributors can still change. Missing interior cells exclude the cohort,
even when its sample endpoints cover the requested exposure.

For a specific pre/post window:

```python
# Choose bounds supported by the fitted sample and estimator.
window = fit.aggregate(balance_window=(-2, 2))
print(window.composition())
```

`balance_window=(lo, hi)` requires every estimated event in that inclusive window
in every retained cohort, then returns only that window. A structurally omitted
`-(anticipation+1)` reference in SA or universal-base CS is exempt from the support requirement
and remains omitted from output. BJS windows start at event `-anticipation` or later. A varying-
base CS pre-period also needs its preceding calendar period, so its support can
be narrower than universal-base CS. No eligible cells is an explicit error.
With [anticipation](anticipation.md), dynamic effects include anticipatory
cells; non-dynamic summaries still start at actual treatment. Contributor and
group-time tables label anticipation separately through `period_type`.

The balancing options are available only for dynamic aggregation and cannot be
combined. They use supported cell estimates, including any earlier omissions
reported by the estimator. `diagnostics["excluded_cohorts"]` lists missing
required event times. `horizons` is applied after the balancing requirement, so
hiding an unsupported horizon does not make a cohort eligible.

**These options balance cohorts, not individual units.** An unbalanced panel
can still contain different observed units at different dates. SA/BJS weights
can also vary across horizons even with a fixed cohort set. Inspect both the
observed counts and `cohort_weights`. Changing composition or balancing can
change the target population; neither is an attrition adjustment.

## BJS storage and validation

Contributor/composition inspection uses only cell-sized metadata and needs no
source-data scan. Saved aggregate tables, and contributor inspection on the
original result, remain available after BJS `cleanup()`. Computing another BJS
aggregate still needs its inference context; weighted and native unbalanced BJS retain
Parquet until cleanup. See [unbalanced panels](unbalanced_panels.md).

Offline fixtures validate R `did::aggte` cohort/calendar vectors, R-style balanced
event studies, and full joint covariance for both control groups, both base
conventions, balanced unconditional and unbalanced conditional panels. Dense
full-unit-dummy score calculations independently validate BJS cohort/calendar
contrasts with every residual grouping, with/without leave-out. Fixture
regeneration is in `benchmarks/external/aggregation_reference.R`.

With [sampling weights](weights.md), cohort/cell weighting uses sampling-weight
totals. `weight_total` reports source-scale totals; `weight_units`,
`observed_units` and `cohort_units` retain physical counts. Composition tables
also report `target_weight_total`, summing the weighting mass of contributing
cells; this can count a unit's weight at several dates.

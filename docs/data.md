# Your data

## The panel

Long form, one row per unit and period: an outcome, a unit identifier, a
period, and a cohort column. Pass a Polars DataFrame or LazyFrame, or a Parquet
path or glob (`"panel/*.parquet"`); only the columns the formula and options
name are read.

- **`panel=(unit, time)`**: the unit and period columns. Unit identifiers may be
  integers or strings. Periods are positive integers, evenly spaced: every
  year, every month, or every second year, but not with irregular gaps.
- **Treatment timing**, given one of two ways:
  - **`cohort=`**: the period each unit is first treated, constant within the
    unit. 0 or null marks a never-treated unit. A cohort after the last period
    is treated as never treated within the sample.
  - **`treatment=`**: a 0/1 (or Boolean) column; each unit's cohort is its
    first period with treatment 1. Treatment must be absorbing: a unit that is
    treated stays treated.
- Each unit-period appears once. Missing or non-finite values in any column
  used are an error: drop or fill them first.

Treatment must start after the first period: a unit treated from the first
period has no untreated observation, and the estimators refuse it (the
[diagnostics](diagnostics.md) accept it).

### Periods that are not consecutive

With evenly spaced periods (2001, 2003, 2005, ...), event times, `anticipation`
and `pretrend_leads` count periods, not years: event time 1 is the next period
after treatment, and `anticipation=1` means one period (two years) before it.
Cohorts and periods keep their labels everywhere else: in coefficient names
(`g::2007:t::2009`), `cohort_time()`, the cohort and calendar aggregates, and
`diagnostics` (`period_step` records the spacing). A cohort must be one of the
panel's periods; for a unit treated between two periods, give the first period
it is observed treated (or pass `treatment=`, which does so).

## Repeated cross sections

When each period is a fresh sample (a survey drawn anew every year), there are
no units to follow. `did_callaway_santanna` accepts such data with `time=`
instead of `panel=`:

```python
fit = did_callaway_santanna(
    "employed ~ age + C(education)", "cps/*.parquet",
    time="year", cohort="state_first_treated",
)
```

The cohort is the period the observation's group (a state, say) is first
treated. Each comparison then uses the observations of the treated and control
cohorts in the two periods compared, as R's did does with `panel = FALSE`; the
standard errors treat the observations as independent. The other estimators
need a panel.

## Units with missing periods

`incomplete_units` says what to do with a unit that lacks some periods:

| `incomplete_units` | |
|---|---|
| `"error"` (default) | refuse the panel |
| `"drop"` | drop those units and fit the balanced panel that remains |
| `"keep"` | use every observed row (an unbalanced panel) |

Dropping changes the population the effects describe, so it is never done
implicitly. With `"keep"`, each estimator uses the rows it can:

- **Callaway-Sant'Anna** compares the units observed in both periods of each
  comparison, using each row's own covariates, as R's did does with
  `allow_unbalanced_panel=TRUE`. A unit's contributions are summed before
  clustering.
- **Sun-Abraham** fits the regression on the observed rows; aggregation
  weights cells by their observed units, as `fixest::sunab` does.
- **BJS, two-stage and ETWFE** fit their untreated model on each unit's
  observed untreated rows. A unit with no untreated row is an error.

Cohort-period cells with no observations are dropped and listed in
`fit.diagnostics["omitted_cells"]`. An unbalanced BJS fit keeps temporary files
for aggregation: call `fit.cleanup()` when done (or use the result as a context
manager, `with did_bjs(...) as fit:`).

## Covariates

The formula is `outcome ~ covariates`, with the formula syntax of hdfe-stream
(and pyfixest): `C(region)` for categories, `I(age**2)` and `log(x)` for
transformations, `x:z` and `x*z` for interactions.

- **Callaway-Sant'Anna**: `"y ~ age + C(region)"`. Covariates are taken from
  the earlier period of each comparison (the base period for post-treatment
  effects), so they may vary over time. Up to 256 design columns.
- **Sun-Abraham, BJS, two-stage**: covariates, and additional fixed effects
  after a bar: `"y ~ age | region^year"` adds region-by-year effects. Unit and
  time fixed effects are always included. Covariates take up to 256 design
  columns. The fixed effects can have any number of levels, absorbed iteratively
  when there are many (`fixed_effect_solver`).
- **Extended TWFE**: covariates, interacted with cohort and time; no
  additional fixed effects.

## Sampling weights

`weights="column"` gives each unit a sampling weight: positive, finite and,
by default, constant within the unit. `allow_time_varying_weights=True` accepts
weights that change over time. Weights enter the estimates, the aggregation and
the standard errors; the number of clusters is still the number of units.
Rescaling all weights by a constant changes nothing.

## Anticipation

`anticipation=A` allows units to respond A periods before treatment: only
periods before `cohort - A` count as untreated, and the last of them is the
reference. Event times stay relative to actual treatment, so the event study
shows the anticipation periods as event times -A to -1.

## When every unit is eventually treated

Without never-treated units, the last cohort to be treated can serve as the
comparison until it is treated. `last_treated_controls` chooses when:

| `last_treated_controls` | |
|---|---|
| `"if_needed"` (default) | use the last-treated cohort only when there are no never-treated units, with a `ReferenceWindowWarning` |
| `"never"` | require never-treated units |
| `"always"` | use the last-treated cohort even when never-treated units exist (they stay controls too) |

The sample then ends before the last cohort's treatment (less anticipation),
and that cohort has no effects of its own. `fit.diagnostics["reference_cohort"]`
and `["reference_cutoff"]` record the choice. This is separate from
`control_group` (Callaway-Sant'Anna and ETWFE), which chooses between
never-treated and not-yet-treated comparison units.

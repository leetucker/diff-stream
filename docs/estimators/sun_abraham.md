# Sun-Abraham

`did_sun_abraham` regresses the outcome on a full set of cohort-by-period
treatment indicators with unit and time fixed effects. Each cohort's effects
are estimated separately, so effects that vary across cohorts do not
contaminate each other; the event study averages them with each cohort's
share of the treated units at each event time. The last period before
treatment (before `cohort - anticipation`) is the reference, and the
comparison units are the never-treated (or a [last-treated
cohort](../data.md#when-every-unit-is-eventually-treated)).

```python
from diff_stream import did_sun_abraham

fit = did_sun_abraham(
    "y ~ x | region^year",
    "panel.parquet",
    panel=("id", "year"),
    cohort="first_treat",
)
fit.aggregate().summary()            # event study
fit.aggregate("overall").summary()
fit.pretrend_test()                  # pre-treatment coefficients jointly zero
```

## The regression

- Without covariates in a balanced panel, each coefficient equals the cohort's
  outcome change since its reference period less the never-treated change, so
  the fit reduces to cohort-period means (it equals Callaway-Sant'Anna with
  never-treated controls and `base_period="universal"`).
- **Covariates** enter linearly and may vary over time: `"y ~ x + z"`.
- **Additional fixed effects** follow a bar: `"y ~ x | region^year"` adds
  region-by-year effects (unit and time effects are always included;
  `"| id + year + region^year"` is the same model), with any number of
  levels (see below). Fixed effects interacted with the unit are not
  supported.
- Unbalanced panels fit the regression on the observed rows; aggregation
  weights cells by their observed treated units, as `fixest::sunab` does.

## Fixed effects with many levels

Additional fixed effects are estimated in one of two ways, chosen by
`fixed_effect_solver`:

- **Dense** (`"dense"`): a dummy for each level joins the regression. It is
  exact and fastest with few levels, but its time grows with the number of
  levels, and it is limited to 256 columns with the covariates.
- **Iterative** (`"iterative"`): the fixed effects are absorbed by
  alternating projections in [hdfe-stream](https://github.com/leetucker/hdfe-stream).
  Its time does not grow with the number of levels, and their number is not
  limited: county-by-year effects with hundreds of thousands of levels work,
  out of core.

The default, `"auto"`, counts the levels in one pass over the data and uses
dense for up to 24 columns (levels plus covariates), iterative above that.
Between 16 and 32 levels the two take about the same time; at 240 levels,
on 8 periods and 50,000 to 200,000 units, dense took 5 to 8 times as long for
Sun-Abraham and 6 to 13 times for BJS. Both give the same estimates and
standard errors (to the solver's tolerance, 1e-10 relative);
`fit.diagnostics["fixed_effect_solver"]` says which was used.

Sun-Abraham's iterative path uses more memory than its dense one when the
levels are few (about 4 GB against 2 to 3 GB on 200,000 units over 8 periods);
BJS's uses less. With a tight `max_memory_gb` and few levels, pass
`fixed_effect_solver="dense"`.

The iterative solver's regression has a column for each cohort-period effect.
When a full batch of rows of that design would not fit in the memory left,
diff-stream passes hdfe-stream smaller batches. The memory limit is still
kept only approximately, because hdfe-stream's sorting pass has buffers of its
own: on 2 million rows with 47 effects, a 2 GB limit peaked at 2.4 GB, and a
3 GB limit was kept. Allow about 0.5 GB above the expected need, or set
`max_memory_gb` a little below the job's allocation.

## Validated against

fixest (`feols` with `sunab()`): coefficients, full covariance and event-time
aggregation, balanced and unbalanced, with weights, covariates, crossed fixed
effects and anticipation, including county-by-year effects with 1,200 levels;
and the public Castle Doctrine panel
(Cheng and Hoekstra 2013). See [Validation](../validation.md).

Sun, L. and Abraham, S. (2021). Estimating dynamic treatment effects in event
studies with heterogeneous treatment effects. *Journal of Econometrics*
225(2), 175-199.

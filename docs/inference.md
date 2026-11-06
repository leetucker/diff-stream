# Inference

Three ways to get standard errors and intervals, from cheapest to most
expensive.

## Analytic, clustered by unit

The default for every estimator. Each unit's contribution to every estimated
part (the effects, the nuisance models, the first stage, and for aggregates
the cohort shares) is summed into an influence function, and the covariance is
the sum of their outer products: standard errors clustered by unit.

- `vcov="CRV0"` (default) or `"CRV1"`, which multiplies the variance by
  N/(N-1) for N units; `fit.with_vcov("CRV1")` switches after fitting.
- Intervals and p values use the normal distribution.
- `confint(joint=True)` gives simultaneous intervals over all reported
  coefficients, from draws of a normal vector with the estimated covariance.

BJS's standard errors are the authors' conservative ones, recomputed for each
aggregate (see [BJS](estimators/bjs.md)); ETWFE's can treat the sample's
covariates and shares as fixed (`treat_sample_as_fixed=True`).

## Multiplier bootstrap (Callaway-Sant'Anna)

R did's default inference for event studies. Instead of refitting, each draw
multiplies every unit's influence function by an independent random weight
(Rademacher by default) and sums. Request it at fit time; aggregates keep the
draws:

```python
fit = did_callaway_santanna(
    "y ~ x", data, panel=("id", "time"), cohort="first_treat",
    multiplier_bootstrap_reps=999, multiplier_bootstrap_seed=42,
)
event_study = fit.aggregate()
event_study.multiplier_confint()             # simultaneous band over the event study
event_study.multiplier_confint(joint=False)  # pointwise
event_study.multiplier_draws()               # the draws
```

The same weight multiplies a unit's share scores, so the bands include the
share uncertainty. `multiplier_bootstrap_weights` chooses `"rademacher"`,
`"mammen"`, `"webb"` or `"norm"`. The cost is one pass over the units; the
draws (repetitions by coefficients) are all that is kept.

## Refit bootstrap (every estimator)

`refit_bootstrap` resamples whole units with replacement and refits the
estimator on each sample, re-estimating its propensity scores, trimming, first
stage and aggregation weights. It costs about `reps` fits, so use it when the
analytic standard errors are in doubt: few clusters, heavy trimming, or a
check on a final specification.

```python
from diff_stream import did_bjs, refit_bootstrap

boot = refit_bootstrap(
    "y ~ x | region^year", "panel.parquet",
    panel=("id", "year"), cohort="first_treat",
    estimator=did_bjs, aggregate="event_time",
    reps=999, seed=42, n_workers=4,
    residual_averaging="cohort",           # any did_bjs option
)
boot.tidy()                         # percentile intervals
boot.confint(method="basic")        # or "normal"
boot.failures()                     # draws that could not be fitted
```

- **`estimator`**: any of the five estimator functions; its options follow as
  keywords. **`aggregate`** bootstraps an aggregate instead of the fitted
  coefficients (not for `did_two_stage`, whose coefficients are reported
  directly); **`coefficients`** selects names.
- The original fit fixes the sample: dropped incomplete units and a
  last-treated cutoff stay as they were. A draw that cannot be fitted (it lost
  a cohort, say) is recorded in `failures()` and never redrawn; standard errors
  and intervals then need `allow_incomplete=True`, since they are conditional
  on the draws that could be fitted.
- Standard errors are the standard deviation of the draws; intervals are
  percentile, basic or normal, pointwise.
- **`n_workers`** runs refits in parallel, each in its own process; the job's
  threads and memory are divided among them. With a fixed `seed` the draws are
  the same serially and in parallel.

## Pre-trend tests

See [Results](results.md#pre-trend-tests).

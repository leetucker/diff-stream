# Callaway-Sant'Anna

`did_callaway_santanna` estimates the average effect for each cohort in each
period, ATT(g, t), by comparing the cohort's outcome change since a base
period with the change among comparison units. Covariates enter by outcome
regression, inverse probability weighting, or the doubly robust combination
of Sant'Anna and Zhao (2020), as in R's did.

```python
from diff_stream import did_callaway_santanna

fit = did_callaway_santanna(
    "y ~ x + C(region)",
    "panel.parquet",
    panel=("id", "time"),
    cohort="first_treat",
    control_group="not_yet_treated",
    adjustment="doubly_robust",
)
fit.cohort_time()                    # ATT(g, t) with sample sizes
fit.aggregate().summary()            # event study
fit.aggregate("overall").summary()   # overall ATT
```

Repeated cross sections are supported: pass `time=` instead of `panel=` (see
[Your data](../data.md#repeated-cross-sections)).

## The comparison

- **`control_group`**: `"never_treated"` (default) compares with units never
  treated; `"not_yet_treated"` adds units treated later, while they are still
  untreated in both periods compared. R's did and this package use the same
  definitions.
- **`base_period`**: post-treatment effects compare with the last period
  before treatment (before `cohort - anticipation`). Pre-treatment
  pseudo-effects compare each period with the one before it (`"varying"`,
  default) or with the same last untreated period (`"universal"`, which omits
  that reference cell, like an event-study regression).

## Covariates

- **`adjustment`**: `"doubly_robust"` (default), `"regression"` (outcome
  regression on the controls) or `"ipw"` (normalized inverse probability
  weighting, with a logit propensity score fitted by maximum likelihood).
- Covariates are taken from the earlier period of each comparison, so
  time-varying covariates are allowed; they are not frozen at one date.
- **`propensity_trim`** (default 0.995): controls with a propensity score above
  it are dropped from IPW and doubly robust comparisons; treated units are
  kept. This is R did's convention.
- The models of all comparisons are fitted together, in shared passes over the
  data; comparisons with the same sample and covariate period share one fit.

Singular designs, perfect separation and comparisons with no controls left
after trimming are errors, not silent drops.

## Effective sample size and balance

A comparison can rest on very few effective controls once propensity weights
are applied. The fit warns (`LowEffectiveSampleWarning`) when a treated or
control group has a Kish effective sample size below `effective_sample_warning`
(default 30; None turns it off).

- `fit.effective_sample_size()` lists them by coefficient and group.
- `fit.suppress(min_effective_sample=30)` reports those coefficients (and
  aggregates using them) as NaN, without changing aggregation weights;
  `min_effective_sample=` at fit time does the same.
- `covariate_balance=True` saves covariate balance tables for each
  comparison, before and after propensity weighting, for
  `fit.covariate_balance()` (standardized differences and variance ratios;
  `balance_distributions=True` adds empirical-distribution comparisons).

## Inference

Standard errors come from the influence functions of every estimated part:
the nuisance models, the comparisons and, for aggregates, the cohort shares
that weight them. `multiplier_bootstrap_reps=999` adds R did's multiplier
bootstrap, for simultaneous confidence bands over an event study
(`fit.aggregate().multiplier_confint()`). See [Inference](../inference.md).

## Validated against

R did (`att_gt`, `aggte`) and DRDID: estimates, full covariance and every
aggregation, for both control groups and base periods, regression, IPW and
doubly robust, balanced and unbalanced panels, weights and anticipation, and
the multiplier bootstrap computed on common draws. The documented `mpdta`
minimum-wage example is reproduced in 24 specifications. See
[Validation](../validation.md).

Callaway, B. and Sant'Anna, P. H. C. (2021). Difference-in-differences with
multiple time periods. *Journal of Econometrics* 225(2), 200-230.
Sant'Anna, P. H. C. and Zhao, J. (2020). Doubly robust
difference-in-differences estimators. *Journal of Econometrics* 219(1),
101-122.

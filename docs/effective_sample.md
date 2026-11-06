# Effective sample size and coefficient suppression

CS emits `LowEffectiveSampleWarning` when a required comparison stratum has
effective sample size (ESS) **below 30**. The default warns and retains all
estimates. The warning threshold is a practical convention, inspired by the
authors' BJS `minn(30)` convention; it is not the same statistic or a guarantee
of reliable inference. BJS/SA support policies are unchanged.

```python
from diff_stream import cs_stream

fit = cs_stream(
    "y ~ x + z",
    "panel.parquet",
    panel=("id", "time"),
    cohort="first_treat",
    weights="sampling_weight",
)
fit.effective_sample_size()  # coefficient, cell, cohort, time, stratum, effective_n
fit.group_time()  # also includes min_effective_n and suppressed
```

The diagnostic uses Kish ESS, `(sum(q))**2 / sum(q**2)`, on nonnegative
comparison weights:

- Treated strata use sampling weights.
- IPW/DR control strata use sampling weights times propensity odds, with
  trimmed controls receiving zero weight.
- Regression controls use sampling weights. This measures sample-weight
  concentration, not the leverage of the outcome regression or its potentially
  signed extrapolation weights.
- Balanced panels report treated/control ESS. Native unbalanced panels report
  treated/control pre/post ESS separately. Each unit occurs once within each
  date-specific stratum; reporting a pooled pre/post ESS would double-count
  units observed on both dates.

The policy takes the minimum across the strata required by each coefficient.
The values are invariant to a common positive scaling of sampling weights.
They do not replace overlap, separation or design-rank checks, and are not
ESS measures of the full nuisance-adjusted influence function.

## Inspect first, then suppress

```python
reported = fit.suppress(min_effective_n=30)
reported.group_time()
reported.aggregate().tidy()

fit.group_time()  # original estimates remain available
```

`suppress()` returns a new result. It keeps coefficient names and cells in
place, setting suppressed estimates and covariance rows/columns to NaN.
Standard errors, p-values and pointwise confidence intervals for those
coefficients are consequently unavailable. Supported estimates and their
joint covariance submatrix are unchanged. No observations are discarded,
nuisance models refitted or coefficients replaced with zero.

The same policy can be requested at fit time:

```python
fit = cs_stream(
    "y ~ x + z",
    "panel.parquet",
    panel=("id", "time"),
    cohort="first_treat",
    ess_warn_threshold=30,
    ess_suppress_threshold=10,
)
```

`ess_suppress_threshold=None` is the default. None or zero disables the warning
when supplied as `ess_warn_threshold`. Thresholds must be finite nonnegative
numbers; suppression uses a strict `<` comparison. Existing suppression is
retained when `suppress()` is called again. Retain the original fit if later
inspection or a less restrictive reporting policy is needed.

This is coefficient suppression, **not refusal of the whole fit**. Genuine
identification failures, nonconvergence and invalid input still raise errors
under the existing estimator rules.

## Aggregation and inference

Suppressing a group-time cell also suppresses every aggregate assigning it
nonzero weight. The aggregate's original weights and target population remain
fixed. Dropping that cell and renormalizing the remaining weights would change
the estimand, so this method does not do so.

```python
# These use the same conservative component-support rule:
a = fit.suppress(30).aggregate("dynamic")
b = fit.aggregate("dynamic").suppress(30)
```

An aggregate's `effective_sample_size()` table retains its contributing cells
and their individual strata. Its suppression criterion is the minimum
component-stratum ESS, **not an estimate of pooled aggregate ESS**. This can
be conservative: pooling could improve precision despite a weak component.
Users can inspect the original unsuppressed aggregate to assess it.

Joint intervals, Wald tests and multiplier bands reject selections containing
suppressed coefficients. Select supported names explicitly to perform those
calculations on their unchanged covariance/draws. Saved multiplier draws mask
suppressed columns; CRV switching and pre-trend selections respect suppression.
Contribution tables retain the original aggregation weights and mask the
reported estimates of suppressed aggregates.

Suppression is a reporting policy. It does not repair weak overlap or adjust
confidence intervals for data-dependent selection. Prefer specifying thresholds
before examining effect estimates. No selective-inference correction is claimed.

The numerical tests compare observed weight ESS, weighted/unweighted and
balanced/native unbalanced paths, threshold boundaries, scaling invariance,
fit-time/post-estimation equivalence, unchanged aggregation weights, covariance,
multiplier draws and supported joint tests.

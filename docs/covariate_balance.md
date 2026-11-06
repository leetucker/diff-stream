# CS covariate-balance diagnostics

CS can save comparison-specific covariate balance at fit time:

```python
from diff_stream import cs_stream

fit = cs_stream(
    "y ~ x + z + C(region)",
    "panel.parquet",
    panel=("id", "time"),
    cohort="first_treat",
    est_method="dr",
    balance=True,  # optional; disabled by default
    balance_ecdf=False,  # optional sorted distribution comparisons
)
fit.covariate_balance()
fit.covariate_balance(cohort=2005, time=2007)
fit.aggregate(horizons=[0, 1]).covariate_balance()
```

Results contain one row per fitted covariate term, cohort/date comparison,
stage and stratum. The intercept is excluded; categorical variables use the
compiled model's indicator coding. Outcome-regression/IPW/DR formulas and
estimation are unchanged by requesting diagnostics. No covariate rows or
propensity scores remain on the result. Calling the method without requesting
`balance=True` at estimation raises an informative error.

## What is compared

- **Before:** the actual treated/control comparison sample, weighted by sampling
  weights when supplied.
- **After (IPW and DR):** treated sampling weights and controls' sampling weights
  multiplied by propensity odds, using the fit's cap and control-only trim.
- **Regression-only:** before diagnostics only. Outcome regression does not
  construct a reweighted covariate sample, so no artificial after table is shown.

Balanced comparisons use covariates and sampling weights from their documented
earlier comparison date. Native unbalanced comparisons report **pre and post
strata separately**, using each observed date's covariates and weights, with the
same pooled propensity model as the fitted comparison. No balanced-unit
intersection is imposed. Last-treated reference restrictions and comparison
eligibility apply before diagnostic construction.

The table includes weighted means/variances, standardized mean differences,
variance ratios, treated/control observation counts and Kish effective sample
sizes. SMD scales are the **before-weighting pooled standard deviations**, held
fixed for the after table. The shared helper's binary conventions apply:
variance is `p(1-p)`, variance ratio is omitted, and both eCDF gaps equal the
absolute mean difference. A zero control variance can yield an infinite ratio;
two zero variances yield an undefined/null ratio. These cases are reported
rather than producing a division error.

Trimmed controls remain in the observation count but receive zero after weight;
ESS reflects those weights. Counts refer to comparison observations, not an
influence-function information measure. These diagnostics describe covariate
balance and overlap; they do not establish parallel trends or validate the
outcome model. Inspect the existing overlap, trimming and ESS diagnostics too.

Aggregated results retain diagnostic rows only for contributing cells. They do
not manufacture a pooled aggregate balance statistic. Tables survive covariance
switches and cleanup. Post-fit coefficient suppression does not reweight samples
or alter their underlying balance diagnostics.

## Computation and validation

The implementation calls `hdfe_stream.core.balance.weighted_balance` and
`effective_sizes`. DiD comparison dates, sample selection and fitted propensity
weights stay in diff-stream. One comparison/stratum is staged to temporary
Parquet at a time and removed after shared-kernel reduction. Additional input
passes are reported in `diagnostics['balance_source_passes']`; diagnostics can
add substantial I/O when many different comparisons are requested.

`balance_ecdf=True` additionally sorts covariate distributions. It can be more
expensive than moment diagnostics; source/sort allocators are not OS-capped by
the estimator's memory guard. Use the common resource launcher for external
process enforcement. The default omits continuous-covariate eCDF calculations.

Tests independently fit weighted propensity scores and verify means, variance,
fixed-scale SMDs, ESS and eCDFs across methods, comparison dates, control groups
and balanced/unbalanced samples. A shared-kernel regression check covers zero
variance, and a portable constrained-resource test combines balance diagnostics
with an all-treated last-reference design.

The local weighted/unbalanced balance probe retained 78,504 units and 317,466
rows and completed under a 1.5 GB kernel cap near 482 MiB RSS. Diagnostics reduced
to 144 output rows. The one-comparison staging policy bounds retained diagnostic
data, while repeated source passes add runtime; see the raw measurements in
`benchmarks/results/reference_balance_resources`. Local kernel checks ran on
Linux/WSL; other platforms use the portable CI fixtures.

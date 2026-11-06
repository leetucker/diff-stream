# Sampling weights

All three estimators accept a sampling-weight column via `weights="column"`.
Weights must be numeric, finite, strictly positive, and constant within each
unit across its observed periods by default. The default `weights=None` gives equal weights.
Balanced, complete and native unbalanced panels support this option, including
anticipation and conditional CS. Set `allow_time_varying_weights=True` to permit
date-specific weights; [the extended weighting guide](richer_models.md) documents
R-compatible CS comparison/share conventions, row-weighted SA/BJS and richer
regressions. Without this explicit opt-in, time-varying weights still raise.
Zero weights remain errors; filter excluded rows explicitly before fitting.

The remaining sections describe the time-invariant unit-weight case. The
extended guide states the changes for observation weights and richer models.

```python
from diff_stream import cs_stream

fit = cs_stream(
    "earnings ~ age + C(region)",
    "panels/*.parquet",
    panel=("person", "year"),
    cohort="first_treat",
    weights="sampling_weight",
    panel_mode="unbalanced",
    anticipation=2,
)
fit.aggregate("simple").summary()
fit.group_time()
fit.contributions()
```

These are sampling weights, separate from propensity weights. They enter the
estimating equations and clustered influence scores. They do not create
replicated observations: the cluster count and CRV1 factor remain based on the
number of retained units, not the sum of weights. Variance uses independent-unit
sampling inference, without survey strata, finite-population corrections or
replicate-weight design features. A common positive rescaling of weights leaves
estimates and standard errors unchanged.

## Estimation and aggregation

CS uses weighted control OLS and weighted unpenalized propensity logit MLE.
Sampling weights multiply both treated and propensity-weighted control
contributions; trimming still applies only to high-score controls. Nuisance
corrections and cross-cell covariance use the weighted scores. Native unbalanced
CS retains observed-date covariates and both dates' observed-row contributions,
folding scores by original unit.

CS aggregation weights cells by the retained cohort's sampling-weight total.
Estimated cohort-share uncertainty includes the random total-weight denominator.
For a cohort total $C_g$, total weight $W$, and unit weight $s_i$, its unnormalized
share score is $s_i(1\{G_i=g\}-C_g/W)$. Using raw weighted indicators without this
centering gives incorrect aggregate covariance when weights vary across units.

SA solves weighted within-cohort calendar-effect systems equivalent to the
fully saturated WLS regression with unit/time effects and cohort-time
interactions. Its covariance clusters weighted regression scores by unit.

BJS fits the untreated first stage by WLS and takes weighted cell means of the
imputation residuals. Smart residual averaging uses the actual contrast weights,
including sampling weights, and leave-out leverage uses the squared totals of
those weights within unit/residual group. All four `avgeffectsby` settings and
`leaveout=True` are supported. SA/BJS aggregation uses the sampling-weight total
of units observed in each target cell, conditioning on the observed treatment
timing and support.

The [anticipation](anticipation.md) and [event-window balancing](aggregation.md)
rules are unchanged. Simple/cohort/calendar aggregates target actual
post-treatment periods; dynamic results can include anticipation periods.

## Counts, normalization and storage

`group_time()` and `contributions()` report `weight_total` on the original input
scale. `n_units`, `weight_units`, `observed_units` and `cohort_units` remain actual
unit counts. Composition's `target_weight_total` sums the weighting mass of
contributing cells, potentially counting a unit's weight at multiple dates.
Diagnostics include the weight column, retained total unit weight, and cohort
weight totals. `counts` is the internal normalized cell/cohort weighting mass;
use the tables for physical counts and input-scale totals.

The snapshot carries one additional numeric column. Weights are scaled by their
input maximum, then normalized to retained-unit mean one for numerical work.
This avoids squaring very large input weights. Whole-unit partitioning checks
constancy; complete-panel deletion recomputes normalization on retained units.
There is no global unit-weight or unit-score table.

Weighted BJS uses the bounded observed-row implementation even for balanced
input. Its result retains temporary Parquet for contrast-specific inference.
Call `fit.cleanup()` after the last aggregation. Stored estimates, covariance
and contributor tables remain available afterward. Equal-weight BJS without a
weight column retains the existing balanced moment shortcut.

## Validation and resources

`tests/test_weights.py` checks R `did::att_gt` estimates, full covariance and
aggregation on balanced/unbalanced samples, intercept-only and conditional
models, regression/IPW/DR, both comparison populations and both base conventions,
with anticipation 0/2. SA uses weighted `fixest::sunab` with unit clustering and
no degrees-of-freedom adjustment. As documented in the anticipation guide,
the installed did 2.5.1 varying-base anticipation corner case uses equivalent
affected-onset labels in reference generation.

BJS full covariance is checked against an independent dense R weighted-unit-dummy
oracle for all averaging and leave-out options. Installed `didimputation` 0.5.1
supplies point-estimate checks only: its weighted variance code multiplies the
design by weights before forming its first-stage Gram, producing squared-weight
rather than WLS-weighted cross-products. The independent oracle uses
$X_0' S_0 X_0$ and weighted cluster scores. Package weighted standard errors are
therefore not asserted to match. Exporters and version notes are retained in
`benchmarks/external/` and `tests/data/weights_reference/`.

Equal-weight equivalence, common rescaling by $10^{-200}$/$10^{200}$,
CRV1, shuffled input, small batches/partitions, complete-panel deletion,
invalid weights and temporary-file cleanup are also checked.

Run the resource smoke check offline with:

```bash
STREAM_LIMITS_MAX_THREADS=2 POLARS_MAX_THREADS=2 \
  .venv/bin/python benchmarks/weights_resources.py
```

It fits and aggregates native unbalanced panels with 100,000 units, 750,000 rows,
two-period anticipation, two threads and a 1 GB RSS watchdog. Local peak RSS was
513.0 MiB for conditional CS, 448.4 MiB for SA and 419.1 MiB for BJS with broad
smart averaging and leave-out. All cleaned up successfully. Results and logs
live in `benchmarks/results/weights_resources/`. This is a smoke check at one
panel shape; effects-squared covariance and design/cohort/period dimensions
still determine numerical-state growth.

BJS [clean-placebo regression](bjs_placebos.md) also supports sampling weights
through `pretrends=L`. Its separate WLS covariance is checked against both dense
R equations and the R package's lead-regression SEs. The weighted post-effect
variance limitation described above does not apply to that separate regression.

The [Goodman–Bacon diagnostic](bacon.md) also accepts `weights="column"` on
balanced/complete panels. It decomposes weighted TWFE with the same weights in
cohort means and comparison population shares, retaining physical counts and
reporting input-scale cohort weight totals separately.

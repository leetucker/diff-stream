# Covariate-adjusted Callaway–Sant'Anna

```python
from diff_stream import cs_stream

fit = cs_stream(
    "y ~ x + z + C(region)",
    "panel.parquet",
    panel=("id", "time"),
    cohort="first_treat",
    est_method="dr",
    control_group="notyettreated",
    base_period="universal",
    trim_level=0.995,
)
fit.group_time()
fit.aggregate().summary()
fit.aggregate("simple").summary()
```

`est_method` accepts `reg` (outcome regression), `ipw` (normalized ATT IPW), or
`dr` (traditional doubly robust estimation, the default). Nuisance fits use
control-sample OLS for outcome changes and unpenalized logit maximum likelihood
for target-cohort membership among eligible target/control units. This matches
[R did](https://bcallaway11.github.io/did/reference/att_gt.html) and its
[DRDID panel routine](https://github.com/pedrohcgs/DRDID/blob/master/R/drdid_panel.R); the alternative improved DR
estimator using inverse-probability tilting is not substituted implicitly.

## Covariate dates and comparison populations

Each group-time effect compares outcomes at t and its selected base date b.
Covariates are evaluated at **min(t,b)**. Thus post-treatment comparisons use
g-1, varying-base pre-treatment comparisons use t-1, and universal-base
pre-treatment comparisons use the earlier calendar date t. Covariates may vary
within units; the package does not freeze all covariates at a single global date.

Never-treated controls remain fixed across times. Not-yet-treated controls are
eligible only if untreated at both comparison dates and outside the target
cohort. Propensity fits are reused only when the target/control population and
covariate date are identical. If every compiled covariate is exactly constant
within units, identical populations can also share a fit across dates.
Outcome regressions with the same design/control population share a Gram solve
for multiple outcome changes. Shared data passes fit independent nuisance
systems together; no full observation-by-effect score matrix is stored.

## Formulas and samples

CS supports the family's numeric expression syntax, interactions, `C()`/`i()`
and categorical columns, with an intercept by default. Numeric expressions use
the existing hdfe-stream expression compiler directly; categorical coding uses
its model-matrix compiler on a small synthetic frame. Levels are discovered in
bounded batches. The current limit is 256 design columns; expansion is guarded
before materializing the full compiled design for a leaf.

Missing/non-finite used values raise errors rather than dropping rows or units.
Singular nuisance designs and non-convergent or separated propensity models
also raise explicit errors. Fixed-effect formula extensions and genuine repeated cross sections remain
later work; positive unit-constant [sampling weights](weights.md) are supported.
[Anticipation](anticipation.md) shifts the clean reference and control eligibility.
[Unbalanced panels](unbalanced_panels.md) use observed-date covariates and
R-compatible repeated-cross-section functionals, clustered by original unit.
BJS and SA still require `y ~ 1`; this formula expansion applies to CS.

## Overlap and trimming

`trim_level=0.995` follows the R DRDID convention: controls with fitted
probability at or above this threshold receive zero weight in the target
contrast. All treated units are retained. Propensity and outcome-regression
nuisance fits still use their full eligible samples, including trimmed controls.
The regression estimator ignores this propensity threshold.

Probabilities are capped at 1-1e-6 for weighting/inference, matching DRDID.
`trim_level=1` disables the upper control cutoff while retaining that cap.
Trimming here is distinct from the symmetric trim bounds offered by the shared
`hdfe_stream.core.ipw` matching/IPW helper. It changes the comparison weighting and can affect
the estimand/identifying assumptions; it is not a cure for absent overlap.
Analytic inference holds trim membership fixed, as does the reference routine.
If no controls remain, the fit raises an error.

Inspect `fit.diagnostics["comparisons"]` for each cached population's covariate
date, target/control counts, propensity range, trimmed/capped counts, effective
control sample size and logit iterations. The summary flags actual trimming or
capping. `nuisance_systems`, `propensity_models`, and `nuisance_source_passes`
report reuse and shared numerical scans, including the final moment/score passes.

## Inference and validation

The analytic unit scores include both outcome-regression and estimated-score
contributions. The result retains joint group-time covariance and score/share
blocks, so dynamic/simple aggregation accounts for estimated cohort shares.
CRV0/CRV1 and normal/joint intervals follow the existing result conventions.

Actual R did/DRDID fixtures check all three methods with numeric and expanded
categorical/interaction formulas, both base conventions, both control
populations, full joint covariance and aggregation. Additional 2×2 references
check explicit control trimming. These checks cover estimator and nuisance
conventions separately from performance measurements.

The temporary shared utilities are listed in the source checkout's
`SHARED_INFRASTRUCTURE.md`. DiD populations, baseline dates and aggregation stay
outside the shared core. `examples/conditional_cs.py` is runnable on simulated data.

Optional [sampling weights](weights.md) enter both the nuisance fits and
ATT score contributions, including estimated cohort-share uncertainty in
aggregation. Sampling weights remain distinct from estimated propensity weights.

The [numerical stress review](numerical_stress.md) extends validation to weak
overlap, wide sampling-weight ranges, near collinearity and extreme changes of
covariate units. It also documents the weighted RMS convergence fix and the
difference between numerical agreement and useful effective sample size.

CS now warns by default below 30 effective observations in a required comparison
stratum. Inspect `effective_sample_size()` and use `suppress(min_effective_n=30)`
for optional post-estimation coefficient masking. An optional fit-time
`ess_suppress_threshold` applies the same policy. See [effective sample size](effective_sample.md)
for weight definitions, conservative aggregate propagation and joint-inference rules.

Request `balance=True` for [comparison-specific covariate balance](covariate_balance.md).
See [reference populations](reference_groups.md) for no-never-treated panels.

# Validation

Every estimator is checked against its authors' implementation and against an
independent calculation: a dense regression with a dummy for every unit, or the
estimator's formulas written out directly. The references are frozen in the
repository (`tests/data/`, with the exporting R scripts in
`benchmarks/external/` and hashes of their inputs, code and outputs), so the
test suite needs neither R nor Stata.

## Against the reference packages

| Estimator | Reference | What is compared |
|---|---|---|
| Callaway-Sant'Anna | R did and DRDID | cohort-period effects, full covariance and all four aggregations; both control groups and base periods; regression, IPW and doubly robust; balanced and unbalanced panels, repeated cross sections (`panel = FALSE`), weights, anticipation, last-treated controls; the multiplier bootstrap on common draws |
| Sun-Abraham | R fixest (`sunab`) | coefficients, full covariance and aggregation; balanced and unbalanced, weights, covariates, crossed fixed effects, anticipation; county-by-period fixed effects with 1,200 levels, absorbed iteratively |
| BJS | the authors' Stata `did_imputation`; R didimputation | estimates and full covariance for every residual averaging group, with and without leave-out (16 baseline and 176 extended Stata fits), including weights, unbalanced panels, anticipation and pre-trend leads; richer covariate and fixed-effect models against a dense regression; county-by-period fixed effects with 1,200 levels, absorbed iteratively, against didimputation (estimates, and standard errors without weights, whose convention differs), including treated observations that cannot be imputed and are dropped |
| Two-stage | the authors' R did2s (pinned source) | coefficients and full covariance in 48 fits: event study and static, weights, anticipation, covariates and crossed fixed effects |
| Extended TWFE | the authors' R etwfe and emfx (pinned source) | cell effects and the covariance of every aggregation, both control groups and fixed-effect choices, conditional and unconditional |
| Stacked | fixest on the stacked data with the corrective weights, built as in the authors' R code | coefficients and full covariance, balanced, unbalanced, weighted and with anticipation |
| Continuous treatment | the authors' R contdid; R did | the overall ATT and ACRT, both dose-response curves and their standard errors, and the event studies of the ATT and the ACRT, with a cubic spline and a quadratic spline with interior knots; not-yet-treated comparisons and a varying base period, the only settings contdid applies. Curves on a dose grid where contdid evaluates them with the basis it fitted, and standard errors with three corrections to contdid and ptetools (see [Differences from other packages](implementations.md#conventions-that-differ)). Its effect of being treated equals Callaway-Sant'Anna exactly; with a linear spline, each cell equals a regression of the outcome change on the dose. A discrete dose against Python's diff-diff 4.0.0 (estimates, both curves and their standard errors, cohort-period effects and the event study, with both comparison groups) and against a direct calculation of every standard error from each unit's influence on the estimate. The lowest-dose comparison (no untreated units) against diff-diff's `control_group="lowest_dose"` (estimates, both curves and their standard errors, cohort-period effects and the event study, for both dose types and base periods), and a group of reference doses against diff-diff with those doses set to one value. Unbalanced panels (experimental) against a direct calculation of every estimate and standard error. Covariates (experimental) against diff-diff's regression adjustment, with a covariate that changes over time and region dummies, for both dose types and comparison groups (estimates, curves and their standard errors, cohort-period effects, event study); the effect of being treated equals Callaway-Sant'Anna's regression adjustment, an absorbed fixed effect equals its dummies, and every standard error equals a direct calculation |
| Switching treatment | the authors' R DIDmultiplegtDYN | estimates, standard errors, numbers of switchers, and the p-values of its joint placebo and equal-effects tests (which use the covariances between the estimates) for effects, placebos and the average total effect, with normalized effects, switchers in or out only, never-switcher comparisons, covariates, weights that change over time, trends by group (`trends_nonparam`) and group-specific linear trends (`trends_lin`), alone and combined, a panel in which every group switches, and an unbalanced panel with and without covariates and weights; with binary absorbing treatment, the effects equal the not-yet-treated Callaway-Sant'Anna event study |
| Goodman-Bacon | R bacondecomp | every comparison's estimate and weight, with and without covariates and weights |
| TWFE weights | the authors' R TwoWayFEWeights (`feTR`) | cell weights and both sensitivity measures, with switching treatment, unbalanced panels, covariates and weights |
| Refit bootstrap | R did on the same resampled units | each refit's estimates, and the intervals from base R |

Estimates and covariances agree to within 2e-7 absolute, and usually to
1e-10; the Stata comparisons, for instance, differ by at most 6e-8 in an
estimate and 3e-9 in a covariance entry. Where a reference uses a different
convention (a small-sample factor, Stata's `minn`), the comparison converts it
explicitly; see [Differences from other packages](implementations.md).

## Published applications

- **mpdta** (county teen employment and minimum wages, 500 counties, a subset
  of Callaway and Sant'Anna's 2021 data built from the Census Bureau's
  Quarterly Workforce Indicators, as distributed with R did): 24
  Callaway-Sant'Anna specifications against R did, with the full covariance of
  every aggregation and the multiplier bootstrap. The extended TWFE example
  uses the same data.
- **Castle Doctrine** (Cheng and Hoekstra 2013, the 50-state panel distributed
  with did2s): Sun-Abraham against fixest in four timing and weighting
  specifications; also the two-stage and Goodman-Bacon checks.
- **math_reform** (Goodman 2019's math graduation requirements, aggregated to
  26 states by 20 cohorts, as distributed with bacondecomp): all 16
  comparisons of the Goodman-Bacon decomposition.
- **Wage panel** (Vella and Verbeek 1998, as distributed with the authors'
  TwoWayFEWeights example): the `feTR` weights of union membership.

Full citations and how each file was exported are in `tests/data/DATA_SOURCES.md`.

## Identities and edge cases

- The iterative fixed-effect solver gives the dense design's estimates and
  covariances (to 1e-10) for Sun-Abraham and BJS, with covariates, weights,
  unbalanced panels, anticipation, leave-out and pre-trend leads.
- Exact identities between the estimators (Sun-Abraham and Callaway-Sant'Anna
  with a universal base; two-stage, ETWFE with not-yet-treated controls, and
  BJS) are tested on every run; see [Choosing an estimator](choosing.md).
- A numerical stress matrix (weak overlap, extreme weights, nearly collinear
  covariates, balanced and unbalanced) compares 80 fits with R did and checks
  that 22 unidentified or separated designs are refused with an error rather
  than returning numbers.
- Callaway-Sant'Anna, Sun-Abraham and BJS give the same results with the panel split into many pieces
  on disk as in one piece, from a DataFrame, a LazyFrame or Parquet files, and
  with shuffled rows and string unit identifiers.

## Under a memory limit

The memory-limit tests (`pytest -m resource`) refit estimators in a child
process under stream-limits' memory watchdog, on Linux, macOS and Windows in
CI, and compare them with an unconstrained fit. On Linux with a delegated
cgroup (`STREAM_LIMITS_TEST_CGROUP_PARENT`) one test runs under a
kernel-enforced limit instead.

## Known limitations

- With NumPy 1.x, the propensity logit of one nearly collinear stress design
  does not meet its convergence test (the Hessian's condition number is about
  3e10, so the Newton steps cannot get below about 1e-7). It converges with
  NumPy 2, which diff-stream requires.

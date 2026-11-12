# Changelog

## Unreleased

- **`did_continuous`**: Callaway, Goodman-Bacon and Sant'Anna's
  difference-in-differences with a continuous treatment. It gives the dose-response
  curves ATT(d) and ACRT(d) (`dose_response()`, with simultaneous bands from the
  multiplier bootstrap), the overall ATT and average causal response, and event
  studies of either (`aggregate(target=...)`). The dose can be continuous (a
  B-spline) or take a few values (`dose_type="discrete"`, a mean for each). When
  every unit is treated at one date, `control_group="lowest_dose"` compares each dose
  with the units at the lowest dose (the paper's Remark 3.1), or with those up to
  `reference_dose`, for effects relative to theirs. Unbalanced panels
  (`incomplete_units="keep"`) are experimental: each comparison uses the units
  observed in both of its periods.
  Covariates and one absorbed fixed effect (`"y ~ x | region"`) are experimental: the
  authors have not developed them, and fits with them issue an `ExperimentalWarning`
  (a subclass of hdfe-stream's) that links to their assumptions. It takes
  staggered adoption, never-treated or not-yet-treated comparisons and both base
  periods, and runs out of core like the other estimators. It matches R's contdid
  except where contdid has known problems, in its dose-response curves and its
  standard errors, and its discrete dose matches Python's diff-diff except in standard
  errors that treat estimated shares as known (see Differences from other packages).
- **`did_dcdh`**: de Chaisemartin and D'Haultfoeuille's event-study estimator
  (`did_multiplegt_dyn`) for a treatment that may take several values and switch on and
  off. It gives the effects l periods after the first switch, placebos and the average
  total effect with their joint covariance, with normalized effects, switchers in or
  out only, comparisons with never-switchers only, covariates, unbalanced panels
  (`incomplete_units="keep"`), weights (also time-varying), trends that differ by
  group-level variables (`trends_by`) and group-specific linear trends
  (`linear_trends`), and tests that the placebos are zero and that the effects are
  equal. It matches the authors' R DIDmultiplegtDYN, except that its covariance matrix
  is a valid one and, with linear trends, it takes no first differences across missing
  periods (see Differences from other packages).

## 0.1.0

The first release.

- **Estimators**, one function each: `did_callaway_santanna` (regression,
  IPW and doubly robust covariate adjustment; never-treated or not-yet-treated
  controls; varying or universal base periods; panels or repeated cross
  sections), `did_sun_abraham`, `did_bjs`
  (the authors' standard errors, residual averaging groups, leave-out, and the
  pre-trend test of `pretrend_leads`), `did_two_stage` (event study or overall
  ATT), `did_extended_twfe` (linear; conditional or unconditional inference)
  and `did_stacked` (stacked DiD with Wing, Freedman and Hollingsworth's
  corrective weights, without building the stack).
  Sun-Abraham, BJS and the two-stage estimator take covariates and additional
  fixed effects (`"y ~ x | region^year"`) and absorb fixed effects with any
  number of levels (county-by-year) through hdfe-stream
  (`fixed_effect_solver`). BJS drops treated observations whose untreated
  outcome cannot be imputed, with an `UnimputedObservationsWarning` and the
  affected cells in its diagnostics.
- **Speed**: Callaway-Sant'Anna with covariates (balanced and unbalanced panels, and
  repeated cross sections), the two-stage estimator, extended TWFE, and Sun-Abraham and BJS with
  covariates run their passes over the data in compiled code, in parallel, as do the
  validation and summing when the panel is first staged, and the next batch of units is read in
  the background. Importing the package sets `OMP_WAIT_POLICY=passive` (unless set), which
  speeds the compiled passes. Results are identical for any number of threads. The first fit after installation compiles for a few seconds, once.
- **Benchmarks** against R's did, fixest, didimputation, did2s, etwfe and
  bacondecomp and Python's diff-diff and pyfixest, from 10,000 to 10 million
  units, with covariates and with fixed effects of millions of levels
  (`docs/benchmarks/`).
- **Data**: Polars frames or Parquet larger than memory; treatment timing as
  a first-treated-period column (`cohort=`, 0 or null for never treated) or a
  0/1 column (`treatment=`); evenly spaced periods (every second year, ...),
  with event times counted in periods and the original labels shown; unbalanced panels
  (`incomplete_units="error" | "drop" | "keep"`), sampling weights (optionally
  time-varying), anticipation, and panels where every unit is eventually
  treated (`last_treated_controls`).
- **Results**: cohort-period effects (`cohort_time()`) and their aggregation
  by event time, cohort, calendar period or overall (`aggregate()`), with
  balanced windows (`balanced_horizon`, `balance_window`) and the cells and
  cohorts behind each estimate (`contributions()`, `composition()`);
  pre-trend tests; effective sample sizes and covariate balance for
  Callaway-Sant'Anna.
- **Inference**: standard errors clustered by unit (CRV0, CRV1) that include
  the uncertainty in aggregation shares; Callaway-Sant'Anna's multiplier
  bootstrap with simultaneous bands; `refit_bootstrap`, which refits any
  estimator on unit resamples, serially or in parallel.
- **Diagnostics**: `bacon_decomp` (with covariates, weights and
  always-treated units) and `twfe_weights` (`feTR`, with switching treatment).
- **Resources**: memory and thread limits from the job (stream-limits);
  staging in memory when the panel fits and in whole-unit pieces on disk
  otherwise, with batch sizes that adapt to the memory left; `profile=True`.
- Validated against R did/DRDID, fixest, didimputation, did2s, etwfe/emfx,
  bacondecomp and TwoWayFEWeights, and the authors' Stata `did_imputation`.

Known limitation: on macOS, the propensity logit of IPW and doubly robust
Callaway-Sant'Anna can fail to converge when two covariates are almost
identical (a condition number near 1e10); it converges on Linux and Windows.
The test for that design is marked as an expected failure on macOS.

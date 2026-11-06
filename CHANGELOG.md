# Changelog

## Unreleased (0.1.0)

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
  fixed effects (`"y ~ x | region^year"`); Sun-Abraham and BJS absorb fixed
  effects with any number of levels (county-by-year) through hdfe-stream
  (`fixed_effect_solver`). BJS drops treated observations whose untreated
  outcome cannot be imputed, with an `UnimputedObservationsWarning` and the
  affected cells in its diagnostics.
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

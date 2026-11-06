# BJS imputation

`did_bjs` fits unit and time fixed effects (and any covariates) on the
untreated observations only: never-treated units, and treated units before
`cohort - anticipation`. It predicts each treated observation's untreated
outcome from that model and averages the differences. Because it uses every
untreated period, it is precise when parallel trends hold over the whole
pre-period.

```python
from diff_stream import did_bjs

fit = did_bjs(
    "y ~ x | region^year",
    "panel.parquet",
    panel=("id", "year"),
    cohort="first_treat",
    pretrend_leads=3,
)
fit.aggregate().summary()            # event study (post-treatment)
fit.aggregate("overall").summary()
fit.pretrend_test()                  # the 3 leads jointly zero
fit.placebos().tidy()                # the lead coefficients
```

## Standard errors

The standard errors are the authors' (as in their Stata command
`did_imputation`): conservative under effect heterogeneity, because the
treated residuals are taken around averages of the effects within groups.

- **`residual_averaging`**: the groups for those averages, `"cohort_time"`
  (default), `"event_time"`, `"cohort"` or `"all"` (the authors'
  `avgeffectsby`). Coarser groups need fewer units per group and are more
  conservative.
- **`leave_out=True`** leaves each unit out of its own group's average (the
  authors' `leaveout`), reducing the downward bias in small groups. A group
  with a single unit is then an error, or NaN for coarser averaging; the
  result's `diagnostics["warnings"]` suggests both options for small cohorts.

The averages are recomputed for each aggregation, so an aggregate's standard
error is not a transformation of the cell covariance.

## Pre-trend test

The imputation model uses all pre-periods, so it has no pre-treatment
effects to test. `pretrend_leads=L` fits a separate regression on the
untreated observations with indicators for the last L periods before
treatment (before `cohort - anticipation`), with the same covariates, fixed
effects and weights. `fit.placebos()` returns it (a `BJSPlaceboResult`, with
`support()` for the units behind each lead) and `fit.pretrend_test()` tests
the leads jointly. The treatment effects are unchanged.

## Covariates and fixed effects

As for Sun-Abraham: `"y ~ x"`, with additional fixed effects after a bar,
`"y ~ x | region^year"`. Fixed effects with many levels, such as
county-by-year, are absorbed iteratively by hdfe-stream
(`fixed_effect_solver`, as described for
[Sun-Abraham](sun_abraham.md#fixed-effects-with-many-levels)); the standard
errors then solve the untreated model's normal equations once per cohort and
period.

### Treated observations that cannot be imputed

A treated observation whose fixed-effect level has no untreated observations
(a county-year in which every unit is already treated) has no untreated
outcome to impute from. Such observations are dropped, with an
`UnimputedObservationsWarning`, and the fit goes on. Dropping them changes no
other observation's imputed effect, since they never enter the untreated
model, but it does change what is estimated: the effects average the
remaining treated observations only. Those are often the late periods of
high-adoption places, so late event times can be affected most.

The warning gives the number of observations dropped and their share of the
treated weight, and `fit.diagnostics` records them: `unimputed_observations`,
`unimputed_treated_weight_share`, and `unimputed_cells`, the cohort-period
cells affected with the observations dropped and remaining in each. A cell that
loses all of its observations is omitted (`omitted_cells`), and aggregation
weights count only the remaining observations. If the share is large,
coarser fixed effects keep more of the treated sample.

R's didimputation drops these observations too, without a warning; Stata's
`did_imputation` stops with an error unless `autosample` is given.

Fits with covariates, additional fixed effects, weights or an unbalanced panel
keep temporary files for aggregation; call `fit.cleanup()` when done, or use
`with did_bjs(...) as fit:`.

## Validated against

The authors' Stata `did_imputation`: 16 baseline fits and 176 extended ones
(every averaging group, with and without leave-out, balanced, unbalanced,
weighted, with anticipation and clean leads), estimates and full covariance.
R didimputation for estimates and standard errors, including county-by-year
fixed effects with 1,200 levels, and an independent dense regression for the
joint covariance of covariate and fixed-effect models. See
[Validation](../validation.md).

Borusyak, K., Jaravel, X. and Spiess, J. (2024). Revisiting event-study
designs: robust and efficient estimation. *Review of Economic Studies*
91(6), 3253-3285.

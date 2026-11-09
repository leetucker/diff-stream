# Two-stage DiD

`did_two_stage` is Gardner's two-stage estimator. The first stage fits unit and
time fixed effects (and any covariates) on the untreated observations; the
second regresses every observation's first-stage residual on treatment
indicators. Its point estimates equal BJS imputation's; its standard errors
treat the two stages as one GMM system, so they account for the estimated
first stage (Butts and Gardner's R package did2s computes the same).

```python
from diff_stream import did_two_stage

event_study = did_two_stage(
    "y ~ x | region^year", "panel.parquet", panel=("id", "year"), cohort="first_treat"
)
event_study.tidy()                         # event::-4 ... event::5
event_study.select([0, 1, 2]).summary()    # a window, with its covariance
event_study.pretrend_test(horizons=[-3, -2])

overall = did_two_stage(
    "y ~ x | region^year", "panel.parquet", panel=("id", "year"),
    cohort="first_treat", event_study=False,
)
overall.summary()                          # one ATT
```

## The specification

- **`event_study=True`** (default): one indicator per event time, omitting the
  reference `reference_period` (default: the last untreated period,
  `-anticipation - 1`). Leads are estimated too, so `pretrend_test()` tests
  them.
- **`event_study=False`**: one indicator for every treated observation (from
  `cohort - anticipation` on), the overall ATT.
- There is no `aggregate()`: the coefficients are the estimands, and the event
  study and the overall ATT are separate fits. `select(horizons)` reports a
  subset without refitting. `support()` lists the cohorts and periods behind
  each coefficient.

Every unit needs at least one untreated observation, and the first stage must
be identified (covariates need variation among the untreated). Covariates and
additional fixed effects follow the Sun-Abraham and BJS syntax.

The additional fixed effects are estimated in one of two ways, chosen by
`fixed_effect_solver`: a dummy for each level in the first-stage regression
(`"dense"`, up to 256 columns with the covariates), or absorbed by alternating
projections in [hdfe-stream](https://github.com/leetucker/hdfe-stream)
(`"iterative"`), whose time does not grow with the number of levels and which
has no limit on them, such as county-by-year. The default, `"auto"`, uses dense
for up to 40 columns and iterative above that; the two give the same estimates
and standard errors, and `fit.diagnostics["fixed_effect_solver"]` says which was
used. [Sun-Abraham](sun_abraham.md#fixed-effects-with-many-levels) gives the
timings, which are about the same for two-stage: at 1 million units over 8
periods, 24 levels took 8 seconds dense against 10 iterative, and 240 levels 48
against 14.

## Validated against

The authors' R `did2s` (from a pinned source commit) and an independent dense
stacked regression: 48 fits, event study and static, balanced and unbalanced,
weights, anticipation, covariates and crossed fixed effects, custom reference
periods, and the Castle Doctrine panel. See [Validation](../validation.md).

Gardner, J. (2022). Two-stage differences in differences. arXiv:2207.05943.
Butts, K. and Gardner, J. (2022). did2s: two-stage difference-in-differences.
*The R Journal* 14(3), 162-173.
